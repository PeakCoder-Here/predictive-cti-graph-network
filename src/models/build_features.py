"""Phase 5 — Feature engineering.

Pulls the whole graph from Neo4j and produces the tensors the GNN needs:
  - x:              [N, F] node features (text embedding + node-type one-hot)
  - node_meta:      list[dict] aligned to row index (label, id, name)
  - edge_index_all: [2, E] all directed edges, as stored in the graph
  - edge_types:     [E]    relationship type per edge (e.g. ASSOCIATED_WITH)
  - edge_sources:   [E]    provenance per edge (only EXPLOITS carries one)

Training (train.py) chooses which relationship type to predict; everything
else becomes the message-passing graph. Saves to data/graph_data.pt.

Run from project root:
    python -m src.models.build_features
"""
from __future__ import annotations

import os
from collections import Counter

import torch
from sentence_transformers import SentenceTransformer

from src.database.db import get_driver

LABELS = ["CVE", "Malware", "ThreatActor", "Vendor", "CWE"]
LABEL_IDX = {l: i for i, l in enumerate(LABELS)}
OUT_PATH = os.path.join("data", "graph_data.pt")


def _fetch_nodes():
    query = """
    MATCH (n)
    RETURN elementId(n) AS eid,
           labels(n)[0]  AS label,
           coalesce(n.id, n.name, '')                AS id,
           coalesce(n.name, n.id, '')                AS name,
           coalesce(n.description, n.name, n.id, '') AS text
    """
    with get_driver().session() as s:
        return [dict(r) for r in s.run(query)]


def _fetch_edges():
    query = """
    MATCH (a)-[r]->(b)
    RETURN elementId(a) AS src, elementId(b) AS dst,
           type(r) AS rel, r.source AS source
    """
    with get_driver().session() as s:
        return [dict(r) for r in s.run(query)]


def build_features():
    print("Fetching nodes and edges from Neo4j...")
    nodes = _fetch_nodes()
    edges = _fetch_edges()
    print(f"  {len(nodes)} nodes, {len(edges)} edges.")

    eid_to_idx = {n["eid"]: i for i, n in enumerate(nodes)}
    node_meta = [{"label": n["label"], "id": n["id"], "name": n["name"]} for n in nodes]

    print("Loading sentence-transformer (first run downloads ~90 MB)...")
    encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    texts = [n["text"][:512] if n["text"] else n["name"] for n in nodes]
    print("Encoding node text (can take a minute on CPU)...")
    emb = torch.tensor(
        encoder.encode(texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True),
        dtype=torch.float,
    )

    type_onehot = torch.zeros((len(nodes), len(LABELS)), dtype=torch.float)
    for i, n in enumerate(nodes):
        type_onehot[i, LABEL_IDX.get(n["label"], 0)] = 1.0
    x = torch.cat([emb, type_onehot], dim=1)

    src, dst, rel, source = [], [], [], []
    for e in edges:
        src.append(eid_to_idx[e["src"]])
        dst.append(eid_to_idx[e["dst"]])
        rel.append(e["rel"])
        source.append(e.get("source") or "")

    data = {
        "x": x,
        "node_meta": node_meta,
        "label_idx": LABEL_IDX,
        "edge_index_all": torch.tensor([src, dst], dtype=torch.long),
        "edge_types": rel,
        "edge_sources": source,
    }

    os.makedirs("data", exist_ok=True)
    torch.save(data, OUT_PATH)
    print(f"\nFeature dim: {x.shape[1]}")
    print("Edge counts by type:", dict(Counter(rel)))
    print(f"Saved graph tensors to {OUT_PATH}")


if __name__ == "__main__":
    build_features()
