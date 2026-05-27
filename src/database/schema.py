"""Schema definition: uniqueness constraints + indexes.

Uses Neo4j 5.x syntax (IF NOT EXISTS), which is idempotent — safe to run
repeatedly. Run this once before any ingestion.
"""
from src.database.db import get_driver

CONSTRAINTS = [
    "CREATE CONSTRAINT threatactor_id IF NOT EXISTS FOR (t:ThreatActor) REQUIRE t.id IS UNIQUE",
    "CREATE CONSTRAINT malware_id     IF NOT EXISTS FOR (m:Malware)     REQUIRE m.id IS UNIQUE",
    "CREATE CONSTRAINT cve_id         IF NOT EXISTS FOR (c:CVE)         REQUIRE c.id IS UNIQUE",
    "CREATE CONSTRAINT cwe_id         IF NOT EXISTS FOR (w:CWE)         REQUIRE w.id IS UNIQUE",
    "CREATE CONSTRAINT vendor_name    IF NOT EXISTS FOR (v:Vendor)      REQUIRE v.name IS UNIQUE",
]

INDEXES = [
    "CREATE INDEX cve_published IF NOT EXISTS FOR (c:CVE) ON (c.published_date)",
    "CREATE INDEX malware_name  IF NOT EXISTS FOR (m:Malware) ON (m.name)",
]


def setup_schema():
    with get_driver().session() as session:
        for stmt in CONSTRAINTS + INDEXES:
            session.run(stmt)
            print(f"  ✓ {stmt.split('IF NOT EXISTS')[0].strip()}")
    print("Schema ready.")


if __name__ == "__main__":
    setup_schema()
