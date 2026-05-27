"""Ingest MITRE ATT&CK (Enterprise) threat actors and malware.

Parses the official STIX 2.1 bundle and loads:
  - intrusion-set  -> (:ThreatActor)
  - malware / tool -> (:Malware)
  - "uses" links    -> (:ThreatActor)-[:ASSOCIATED_WITH]->(:Malware)

The STIX bundle does not include an actor's country of origin, so we leave
origin_country unset rather than inventing it.

Source: https://github.com/mitre-attack/attack-stix-data
"""
from __future__ import annotations

import requests

from src.database.db import get_driver

STIX_URL = (
    "https://raw.githubusercontent.com/mitre-attack/attack-stix-data/"
    "master/enterprise-attack/enterprise-attack.json"
)


def _mitre_id(obj: dict) -> str | None:
    """Pull the human-facing ATT&CK ID (e.g. G0016, S0154) from STIX refs."""
    for ref in obj.get("external_references", []):
        if ref.get("source_name") == "mitre-attack":
            return ref.get("external_id")
    return None


def _is_active(obj: dict) -> bool:
    return not obj.get("revoked") and not obj.get("x_mitre_deprecated")


def parse_stix() -> tuple[list[dict], list[dict], list[dict]]:
    print("Fetching MITRE ATT&CK STIX bundle (~9 MB)...")
    objects = requests.get(STIX_URL, timeout=180).json()["objects"]

    actors, malware, links = {}, {}, []
    actor_refs, malware_refs = set(), set()

    for o in objects:
        if o["type"] == "intrusion-set" and _is_active(o):
            mid = _mitre_id(o)
            if mid:
                actors[o["id"]] = {
                    "id": mid,
                    "name": o["name"],
                    "aliases": o.get("aliases", []),
                    "description": o.get("description", ""),
                }
                actor_refs.add(o["id"])
        elif o["type"] in ("malware", "tool") and _is_active(o):
            mid = _mitre_id(o)
            if mid:
                malware[o["id"]] = {
                    "id": mid,
                    "name": o["name"],
                    "type": "tool" if o["type"] == "tool" else "malware",
                    "description": o.get("description", ""),
                }
                malware_refs.add(o["id"])

    for o in objects:
        if (
            o["type"] == "relationship"
            and o.get("relationship_type") == "uses"
            and o["source_ref"] in actor_refs
            and o["target_ref"] in malware_refs
        ):
            links.append(
                {
                    "actor_id": actors[o["source_ref"]]["id"],
                    "malware_id": malware[o["target_ref"]]["id"],
                }
            )

    actors_list = list(actors.values())
    malware_list = list(malware.values())
    print(
        f"  Parsed {len(actors_list)} actors, "
        f"{len(malware_list)} malware/tools, {len(links)} links."
    )
    return actors_list, malware_list, links


def ingest_mitre():
    actors, malware, links = parse_stix()
    with get_driver().session() as session:
        session.run(
            """
            UNWIND $rows AS row
            MERGE (a:ThreatActor {id: row.id})
              SET a.name = row.name, a.aliases = row.aliases,
                  a.description = row.description
            """,
            rows=actors,
        )
        session.run(
            """
            UNWIND $rows AS row
            MERGE (m:Malware {id: row.id})
              SET m.name = row.name, m.type = row.type,
                  m.description = row.description
            """,
            rows=malware,
        )
        session.run(
            """
            UNWIND $rows AS row
            MATCH (a:ThreatActor {id: row.actor_id})
            MATCH (m:Malware {id: row.malware_id})
            MERGE (a)-[:ASSOCIATED_WITH]->(m)
            """,
            rows=links,
        )
    print("MITRE ATT&CK ingestion complete.")


if __name__ == "__main__":
    ingest_mitre()
