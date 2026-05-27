"""Phase 3 — Exploratory graph analytics.

Runs the high-value Cypher queries that prove the graph's structure and
surface insights before any machine learning. Run from project root:

    python -m src.database.analytics
"""
from __future__ import annotations

from src.database.db import get_driver


def _run(title: str, query: str, **params):
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")
    with get_driver().session() as s:
        rows = [dict(r) for r in s.run(query, **params)]
    if not rows:
        print("  (no results)")
        return rows
    headers = list(rows[0].keys())
    widths = {h: max(len(h), *(len(str(r[h])) for r in rows)) for h in headers}
    print("  " + "  ".join(h.ljust(widths[h]) for h in headers))
    print("  " + "  ".join("-" * widths[h] for h in headers))
    for r in rows:
        print("  " + "  ".join(str(r[h]).ljust(widths[h]) for h in headers))
    return rows


def graph_summary():
    _run(
        "GRAPH SUMMARY — node counts by label",
        """
        MATCH (n)
        UNWIND labels(n) AS label
        RETURN label, count(*) AS count
        ORDER BY count DESC
        """,
    )
    _run(
        "EXPLOITS edge provenance (real vs heuristic)",
        """
        MATCH ()-[e:EXPLOITS]->()
        RETURN e.source AS source, count(*) AS edges,
               round(avg(e.confidence), 2) AS avg_confidence
        ORDER BY edges DESC
        """,
    )


def most_targeted_vendors(limit: int = 10):
    _run(
        f"MOST-TARGETED VENDORS (top {limit})",
        """
        MATCH (v:Vendor)<-[:AFFECTS]-(c:CVE)<-[:EXPLOITS]-(m:Malware)
        RETURN v.name AS vendor,
               count(DISTINCT m) AS malware_count,
               count(DISTINCT c) AS exploited_cves
        ORDER BY malware_count DESC, exploited_cves DESC
        LIMIT $limit
        """,
        limit=limit,
    )


def most_active_actors(limit: int = 10):
    _run(
        f"MOST-EQUIPPED THREAT ACTORS by malware arsenal (top {limit})",
        """
        MATCH (a:ThreatActor)-[:ASSOCIATED_WITH]->(m:Malware)
        RETURN a.name AS actor, count(DISTINCT m) AS malware_count
        ORDER BY malware_count DESC
        LIMIT $limit
        """,
        limit=limit,
    )


def attack_paths(actor_substring: str = "APT29", limit: int = 10):
    _run(
        f"ATTACK PATHS for actors matching '{actor_substring}' (top {limit})",
        """
        MATCH (a:ThreatActor)-[:ASSOCIATED_WITH]->(m:Malware)
              -[:EXPLOITS]->(c:CVE)-[:AFFECTS]->(v:Vendor)
        WHERE toLower(a.name) CONTAINS toLower($sub)
        RETURN a.name AS actor, m.name AS malware,
               c.id AS cve, v.name AS vendor
        LIMIT $limit
        """,
        sub=actor_substring,
        limit=limit,
    )


def top_weaknesses(limit: int = 10):
    _run(
        f"MOST COMMON WEAKNESS CLASSES — CWE (top {limit})",
        """
        MATCH (c:CVE)-[:UNDER_CLASS]->(w:CWE)
        RETURN w.id AS cwe, count(c) AS cve_count
        ORDER BY cve_count DESC
        LIMIT $limit
        """,
        limit=limit,
    )


def main():
    graph_summary()
    most_targeted_vendors()
    most_active_actors()
    top_weaknesses()
    attack_paths("APT29")


if __name__ == "__main__":
    main()
