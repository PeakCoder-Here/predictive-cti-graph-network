"""End-to-end ingestion runner.

Usage (from project root):
    python -m src.ingestion.run_all

Order matters: schema first, then nodes, then linking.
"""
from src.database.schema import setup_schema
from src.ingestion.ingest_mitre import ingest_mitre
from src.ingestion.ingest_cisa_kev import ingest_kev
from src.ingestion.derive_exploits import derive_exploits


def main():
    print("=" * 50)
    print("STEP 1/4  Schema")
    print("=" * 50)
    setup_schema()

    print("\n" + "=" * 50)
    print("STEP 2/4  MITRE ATT&CK (actors + malware)")
    print("=" * 50)
    ingest_mitre()

    print("\n" + "=" * 50)
    print("STEP 3/4  CISA KEV (CVEs + vendors)")
    print("=" * 50)
    ingest_kev()

    print("\n" + "=" * 50)
    print("STEP 4/4  Derive EXPLOITS edges (prediction target)")
    print("=" * 50)
    derive_exploits()

    print("\nAll ingestion complete. Open http://localhost:7474 to explore.")


if __name__ == "__main__":
    main()
