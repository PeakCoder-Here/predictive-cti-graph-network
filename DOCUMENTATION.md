# Predictive CTI Graph Network — Documentation

A cyber threat intelligence (CTI) knowledge graph and Graph Neural Network (GNN)
system that models the relationships between threat actors, malware,
vulnerabilities, and vendors, and predicts likely links between them.

Built entirely on **real, free, no-API-key data** from MITRE ATT&CK and the
CISA Known Exploited Vulnerabilities catalog.

---

## Table of contents

1. [What this project does](#1-what-this-project-does)
2. [The honest story behind the design](#2-the-honest-story-behind-the-design)
3. [System architecture](#3-system-architecture)
4. [The knowledge graph](#4-the-knowledge-graph)
5. [Data sources](#5-data-sources)
6. [Technology stack](#6-technology-stack)
7. [Prerequisites](#7-prerequisites)
8. [Installation and setup](#8-installation-and-setup)
9. [Usage](#9-usage)
10. [How the GNN works](#10-how-the-gnn-works)
11. [Results](#11-results)
12. [Project structure](#12-project-structure)
13. [Limitations and honest caveats](#13-limitations-and-honest-caveats)
14. [Future work](#14-future-work)

---

## 1. What this project does

Threat intelligence is fundamentally relational: a threat actor *uses* certain
malware, that malware *exploits* certain vulnerabilities, and those
vulnerabilities *affect* certain vendors. This is a graph, not a table — which
makes it a natural fit for graph databases and graph machine learning.

This project ingests real-world CTI into a Neo4j graph database, then trains
Graph Neural Networks to perform **link prediction**: given the existing
structure of the graph, which *missing* links are most likely to be real? In
practical terms, the system answers questions like:

- *Given a threat actor's known profile, which other malware or tools are they
  likely to also use?*
- *Which vulnerabilities is a given actor's toolkit most likely associated with?*

The whole system is presented through an interactive Streamlit dashboard where
you can select any of 174 threat actors, see their attack subgraph rendered as
an interactive network, and view live model predictions.

---

## 2. The honest story behind the design

This section matters more than any metric, because it is what separates a
credible engineering project from a demo that inflates its own results.

The original concept was to predict **malware → CVE** links — i.e. forecast
which vulnerabilities a piece of malware would exploit. While building the
ingestion pipeline, a hard reality surfaced: **there is no clean, free,
structured dataset that maps malware to the CVEs it exploits at scale.** When
the available real sources were mined exhaustively:

- Scanning every CISA KEV vulnerability description for mentions of known
  malware names yielded only **9** confident links.
- Scanning every MITRE ATT&CK malware and group description for CVE references
  yielded only **4** more.

Thirteen real edges is not enough to train or even meaningfully evaluate a
machine learning model. Rather than fabricate data and report an impressive but
meaningless accuracy figure, the project takes two honest paths simultaneously:

1. **A real headline model** predicting the `ASSOCIATED_WITH` relationship
   (threat actor → malware), for which MITRE provides **1,145 genuine,
   curated edges**. This is the scientifically defensible result.

2. **An experimental model** predicting the `EXPLOITS` relationship
   (malware → CVE), trained on heuristic edges derived from CISA's ransomware
   flags. Its high score reflects the model re-learning the heuristic, **not**
   genuine discovery — and every place it appears is labeled as such.

This transparency is a deliberate design choice. A model that reports
ROC-AUC 0.68 on real labels is worth far more than one reporting 0.99 on
labels it essentially invented.

---

## 3. System architecture

The system has four layers, each isolated in its own package:

**Ingestion layer** pulls raw data from MITRE ATT&CK and CISA KEV over HTTP,
parses it, and loads it into Neo4j as nodes and relationships. It also derives
the experimental `EXPLOITS` edges and tags each with its provenance.

**Database layer** is a Neo4j 5.26 instance running in Docker, with the Graph
Data Science plugin enabled. It stores the knowledge graph and serves both the
analytics queries and the ML feature extraction.

**Machine learning layer** extracts the graph into tensors, encodes every
node's text into embeddings, trains the GNNs, and runs inference. It is built on
PyTorch Geometric.

**Presentation layer** is a Streamlit dashboard that queries Neo4j for live
graph data and calls the trained models for predictions, rendering everything in
an interactive web interface.

Data flows in one direction during setup (sources → ingestion → Neo4j → ML
features → trained models) and the dashboard reads from both Neo4j and the
trained models at runtime.

---

## 4. The knowledge graph

The graph schema connects five node types through four relationship types:

```
(:ThreatActor)-[:ASSOCIATED_WITH]->(:Malware)-[:EXPLOITS]->(:CVE)-[:AFFECTS]->(:Vendor)
                                                            (:CVE)-[:UNDER_CLASS]->(:CWE)
```

The node types are: **ThreatActor** (an APT group or named adversary),
**Malware** (a malware family or offensive tool), **CVE** (a specific
vulnerability), **Vendor** (the affected software vendor or product), and
**CWE** (the weakness class a vulnerability belongs to).

After ingestion the live graph contains roughly **3,034 nodes** and **5,255
relationships**, broken down as 174 threat actors, 821 malware/tools, plus the
CVEs, vendors, and weakness classes from CISA, connected by 1,145
ASSOCIATED_WITH edges, 1,602 AFFECTS edges, 1,530 UNDER_CLASS edges, and 978
(heuristic) EXPLOITS edges.

Each `EXPLOITS` edge carries two properties — `source` (either `text_match` or
`ransomware_affinity`) and `confidence` — so that training can filter to
real-only edges or include the heuristic ones with a single flag.

---

## 5. Data sources

**MITRE ATT&CK (Enterprise)** is the authoritative open knowledge base of
adversary tactics and techniques. The project consumes the official STIX 2.1
bundle directly from MITRE's GitHub repository. From it we extract intrusion
sets (threat actors), malware and tools, and the curated "uses" relationships
linking actors to their software. No API key is required.

**CISA Known Exploited Vulnerabilities (KEV)** is the U.S. government's
authoritative list of vulnerabilities confirmed to be exploited in the wild.
The project consumes the single JSON feed published by CISA. Each entry provides
a CVE identifier, the affected vendor and product, the associated weakness (CWE)
classes, and a flag indicating known use in ransomware campaigns. No API key is
required.

Both feeds are public, free, and refreshed regularly, so re-running ingestion
picks up newly disclosed threats automatically.

---

## 6. Technology stack

The database is **Neo4j 5.26 Community** with the Graph Data Science plugin,
run via Docker Compose. The ingestion and ML code is **Python 3.13**. Graph
machine learning uses **PyTorch** and **PyTorch Geometric**. Node text is
embedded with the **sentence-transformers** model `all-MiniLM-L6-v2`, which
produces 384-dimensional embeddings. Evaluation metrics come from
**scikit-learn**. The dashboard is built with **Streamlit**, and the
interactive network visualization uses **Pyvis** (a Python wrapper over
vis-network).

---

## 7. Prerequisites

Running this project requires Docker Desktop (with the WSL 2 backend on
Windows), Python 3.10 or newer, and roughly 4 GB of free RAM — Neo4j needs about
1 GB and the ML stack needs the rest. A working internet connection is required
during initial ingestion to download the data feeds and the embedding model.

This stack is **not** suitable for low-RAM single-board computers or phones;
PyTorch Geometric in particular does not build cleanly on ARM/Termux
environments.

---

## 8. Installation and setup

### Step 1 — Start the database

From the project root:

```bash
docker compose up -d
```

The first run downloads the Neo4j image (~500 MB). After about 30 seconds,
confirm the database is live by opening `http://localhost:7474` in a browser.
Log in with username `neo4j` and the password from your `.env` file (the default
is `SecurePassword123`).

### Step 2 — Create the Python environment

```bash
python -m venv .venv
# Windows:
.venv\Scripts\Activate.ps1
# macOS / Linux:
source .venv/bin/activate
```

Your shell prompt should now show `(.venv)`.

### Step 3 — Configure credentials

```bash
cp .env.example .env      # Windows: copy .env.example .env
```

Edit `.env` if you changed the Neo4j password. This file is gitignored and never
committed.

### Step 4 — Install dependencies

For ingestion and the dashboard, the lightweight packages are enough:

```bash
pip install neo4j requests python-dotenv pandas streamlit pyvis
```

For the machine learning layer, install the full set (PyTorch is a large
download, several hundred MB):

```bash
pip install -r requirements.txt
```

> **Note for Python 3.13 users:** PyTorch only supports Python 3.13 from version
> 2.6 onwards, which is why `requirements.txt` specifies `torch>=2.6`.

---

## 9. Usage

All commands are run from the project root with the virtual environment active
and Neo4j running.

### Ingest the data

```bash
python -m src.ingestion.run_all
```

This runs the full pipeline in order: schema setup, MITRE ingestion, CISA KEV
ingestion, and EXPLOITS edge derivation. It takes a couple of minutes. When it
finishes, refresh the Neo4j browser and you should see thousands of nodes.

### Explore with analytics

```bash
python -m src.database.analytics
```

Prints the most-targeted vendors, the most heavily equipped threat actors, the
most common weakness classes, sample attack paths, and the EXPLOITS edge
provenance breakdown.

### Build features and train

```bash
python -m src.models.build_features    # graph -> tensors + text embeddings
python -m src.models.train_all          # trains both GNN models
```

`build_features` downloads the embedding model on first run (~90 MB) and saves
`data/graph_data.pt`. `train_all` trains the real `ASSOCIATED_WITH` model and
the experimental `EXPLOITS` model, printing ROC-AUC during training and a final
test score for each.

To train a single model with options:

```bash
python -m src.models.train --target ASSOCIATED_WITH --epochs 200
python -m src.models.train --target EXPLOITS --sources all
```

### Run predictions from the command line

```bash
python -m src.models.predict "APT29"
```

Prints both the real malware-association predictions and the experimental CVE
predictions for the named actor.

### Launch the dashboard

```bash
streamlit run src/ui/dashboard.py
```

Opens at `http://localhost:8501`. Use the sidebar to select a threat actor and
adjust the number of predictions, then explore the four tabs: Predictions (both
models side by side), Subgraph (interactive attack graph), Analytics
(most-targeted vendors and relationship breakdown), and About (methodology).

---

## 10. How the GNN works

The model is a two-stage architecture: an **encoder** that learns a vector
representation for every node, and a **decoder** that scores whether two nodes
should be linked.

The encoder is a two-layer Graph Convolutional Network (GCN). Each node starts
with a feature vector of dimension 389 — a 384-dimensional sentence embedding of
the node's text (its MITRE description, CVE summary, etc.) concatenated with a
5-dimensional one-hot encoding of its type. The first GCN layer aggregates
information from each node's immediate neighbours; the second layer extends this
to the two-hop neighbourhood. The result is an embedding for each node that
captures both its own content and its structural context in the graph.

The decoder takes the learned embeddings of two candidate nodes, concatenates
them, and passes them through a small multilayer perceptron that outputs a
single score — the predicted probability that an edge should exist between them.

Training treats the target relationship's real edges as positive examples and
randomly sampled non-edges (between valid endpoint types) as negative examples,
optimizing binary cross-entropy. The target edges are split 70/15/15 into
train/validation/test sets; crucially, only the training edges are visible to
the encoder during message passing, so the model is genuinely predicting
held-out links rather than memorizing them.

---

## 11. Results

| Model | Relationship | Edges | Source | Test ROC-AUC | Test AP |
|-------|-------------|-------|--------|-------------|---------|
| **Real (headline)** | ASSOCIATED_WITH (actor → malware) | 1,145 | Real, MITRE | **0.68** | **0.74** |
| Experimental | EXPLOITS (malware → CVE) | 978 | Heuristic | 0.99 | 0.97 |

The **ASSOCIATED_WITH** result is the genuine contribution. A ROC-AUC of 0.68 on
held-out real edges means the model reliably ranks true actor–malware
associations above random pairs, learning meaningful patterns from a sparse
real-world graph. The small gap between validation (~0.71–0.79) and test (0.68)
indicates a reasonably generalizing model rather than one that has overfit.

The **EXPLOITS** model's 0.99 is intentionally not the headline. Because its
training labels come from a deterministic heuristic (ransomware-flagged CVEs
linked to ransomware malware), a near-perfect score simply confirms the model
learned that heuristic — it does not represent discovery of unknown real-world
exploits. It is retained as a forward-looking prototype, clearly captioned
throughout the system.

Interestingly, when queried for APT29, the experimental model surfaced several
Fortinet vulnerabilities (e.g. CVE-2018-13379, CVE-2022-40684) that have in fact
been attributed to APT29-linked activity in real threat reporting — a sign that
even the heuristic graph carries some genuine signal, though this should be
treated as anecdotal rather than validated.

---

## 12. Project structure

```
Predictive-CTI-GraphNetwork/
├── docker-compose.yml          # Neo4j service definition
├── requirements.txt            # Python dependencies
├── .env.example                # Credentials template (copy to .env)
├── .gitignore                  # Excludes .env, .venv, data, models
├── README.md                   # Quick start
├── DOCUMENTATION.md            # This file
├── LICENSE                     # MIT
└── src/
    ├── database/
    │   ├── db.py               # Neo4j connection helper
    │   ├── schema.py           # Constraints and indexes
    │   └── analytics.py        # Exploratory Cypher queries
    ├── ingestion/
    │   ├── ingest_mitre.py     # MITRE ATT&CK -> actors + malware
    │   ├── ingest_cisa_kev.py  # CISA KEV -> CVEs + vendors + CWEs
    │   ├── derive_exploits.py  # Heuristic EXPLOITS edges (tagged)
    │   └── run_all.py          # Full ingestion pipeline
    ├── models/
    │   ├── gnn.py              # GCN encoder + MLP decoder
    │   ├── build_features.py   # Graph -> tensors + embeddings
    │   ├── train.py            # Training for any relationship type
    │   ├── train_all.py        # Trains both models
    │   └── predict.py          # Inference for both models
    └── ui/
        └── dashboard.py        # Streamlit dashboard
```

Note that `data/` (graph tensors and trained models) and `.venv/` are not
committed — they are regenerated by running the pipeline. This keeps the
repository lightweight.

---

## 13. Limitations and honest caveats

The single biggest limitation is the absence of real malware → CVE training
data, discussed at length in section 2. The EXPLOITS model should be regarded as
a prototype illustrating the pipeline, not a validated predictor.

The graph captures only the relationships present in MITRE and CISA. Threat
actor attribution is inherently uncertain and these sources, while
authoritative, are not exhaustive — many real associations are absent simply
because they have not been publicly documented and curated.

Node features rely on text descriptions and node type. Behavioral features (such
as the specific ATT&CK techniques an actor uses) are not yet incorporated, which
limits how much the model can distinguish actors with similar descriptions.

Finally, ROC-AUC and average precision measure ranking quality on held-out
edges; they do not constitute operational validation. This is a research and
learning project, not a production threat intelligence tool.

---

## 14. Future work

The most impactful next step would be enriching node features with MITRE ATT&CK
techniques — adding the attack-pattern nodes and the technique relationships
would give the GNN behavioral signal beyond text, which should improve the real
model meaningfully.

Other directions include experimenting with more expressive GNN architectures
(GraphSAGE or Graph Attention Networks in place of the plain GCN), incorporating
temporal information so the model can reason about how threats evolve over time,
and integrating additional free sources to densify the graph.

---

*This project was built as a hands-on exploration of graph databases and graph
machine learning applied to cyber threat intelligence. Its guiding principle is
intellectual honesty: report what the data can actually support, and be explicit
about what it cannot.*
