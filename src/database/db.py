"""Neo4j connection helper.

Loads credentials from the environment (.env) and exposes a single shared
driver. Import `get_driver()` anywhere you need database access.
"""
from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv
from neo4j import GraphDatabase, Driver

load_dotenv()


@lru_cache(maxsize=1)
def get_driver() -> Driver:
    """Return a process-wide singleton Neo4j driver."""
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "SecurePassword123")
    driver = GraphDatabase.driver(uri, auth=(user, password))
    driver.verify_connectivity()
    return driver


def run_query(query: str, **params):
    """Convenience wrapper: run a single query and return a list of records."""
    with get_driver().session() as session:
        return list(session.run(query, **params))


if __name__ == "__main__":
    # Quick connectivity smoke test
    rows = run_query("RETURN 'Neo4j connection OK' AS status")
    print(rows[0]["status"])
