# Predictive CTI Graph Network

A Cyber Threat Intelligence knowledge graph + Graph Neural Network for
predicting which threat actors are likely to exploit which vulnerabilities.

Built on **real, free, no-API-key data**: MITRE ATT&CK and CISA KEV.

## Status

- [x] Phase 1 — Neo4j infrastructure (Docker)
- [x] Phase 2 — Ingestion / ETL (MITRE ATT&CK + CISA KEV)
- [x] Phase 3 — Graph analytics (Cypher)
- [x] Phase 4 — EXPLOITS edge derivation (text-mining + ransomware heuristic, tagged)
- [x] Phase 5 — GNN link prediction (PyTorch Geometric)
- [x] Phase 6 — Streamlit dashboard

## Prerequisites

- Docker + Docker Compose
- Python 3.10+
- ~4 GB free RAM (for Neo4j + ML). **Not suitable for low-RAM devices.**

## Setup

```bash
# 1. Start Neo4j
docker compose up -d
# wait ~30s, then confirm http://localhost:7474 loads (login neo4j / SecurePassword123)

# 2. Python environment
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3. Configure secrets
cp .env.example .env        # edit if you changed the Neo4j password

# 4. Run the full ingestion (from project root)
python -m src.ingestion.run_all
```

## Machine learning: two link-prediction models

Public CTI data has almost no real malware→CVE links, so we train two models:

1. **ASSOCIATED_WITH** (ThreatActor → Malware) — ~1,145 **real** edges from
   MITRE. Headline model with trustworthy metrics. Answers: *given an actor's
   profile, which other malware/tools are they likely to use?*
2. **EXPLOITS** (Malware → CVE) — heuristic edges (see Phase 4). **Experimental
   only**; results are caveated, not ground truth.

```bash
python -m src.models.build_features      # graph -> tensors (+ embeddings)
python -m src.models.train_all           # trains both models
python -m src.models.predict "APT29"     # predictions from both
```

## Data model

```
(:ThreatActor)-[:ASSOCIATED_WITH]->(:Malware)-[:EXPLOITS]->(:CVE)-[:AFFECTS]->(:Vendor)
                                                              (:CVE)-[:UNDER_CLASS]->(:CWE)
```

Sources:
- **MITRE ATT&CK** → ThreatActor, Malware, ASSOCIATED_WITH
- **CISA KEV** → CVE, Vendor, CWE, AFFECTS, UNDER_CLASS

## Project layout

```
src/
  database/    db connection + schema
  ingestion/   ETL from MITRE & CISA
  models/      GNN (coming)
  ui/          Streamlit dashboard (coming)
```
