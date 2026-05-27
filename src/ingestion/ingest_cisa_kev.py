"""Ingest the CISA Known Exploited Vulnerabilities (KEV) catalog.

This is the single best free, no-key source for *confirmed exploited* CVEs.
Each entry gives us a CVE, the affected vendor/product, and often whether it
has been used in ransomware campaigns. We use these as ground-truth nodes.

Source: https://www.cisa.gov/known-exploited-vulnerabilities-catalog
Feed:   https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json
"""
from __future__ import annotations

import requests

from src.database.db import get_driver

KEV_FEED = (
    "https://www.cisa.gov/sites/default/files/feeds/"
    "known_exploited_vulnerabilities.json"
)

# One Cypher statement, batched with UNWIND for speed.
INGEST_QUERY = """
UNWIND $rows AS row
MERGE (c:CVE {id: row.cveID})
  SET c.description    = row.shortDescription,
      c.published_date = date(row.dateAdded),
      c.vuln_name      = row.vulnerabilityName,
      c.ransomware_use = row.knownRansomwareCampaignUse
MERGE (v:Vendor {name: row.vendorProject})
  SET v.category = 'Unknown'
MERGE (c)-[:AFFECTS]->(v)
WITH c, row
WHERE row.cwes IS NOT NULL AND size(row.cwes) > 0
UNWIND row.cwes AS cweId
MERGE (w:CWE {id: cweId})
MERGE (c)-[:UNDER_CLASS]->(w)
"""


def fetch_kev() -> list[dict]:
    print("Fetching CISA KEV catalog...")
    resp = requests.get(KEV_FEED, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    vulns = data.get("vulnerabilities", [])
    print(f"  Retrieved {len(vulns)} known-exploited vulnerabilities.")
    return vulns


def ingest_kev(batch_size: int = 500):
    vulns = fetch_kev()
    with get_driver().session() as session:
        for i in range(0, len(vulns), batch_size):
            batch = vulns[i : i + batch_size]
            session.run(INGEST_QUERY, rows=batch)
            print(f"  Ingested {min(i + batch_size, len(vulns))}/{len(vulns)}")
    print("CISA KEV ingestion complete.")


if __name__ == "__main__":
    ingest_kev()
