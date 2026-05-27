"""Phase 6 — Streamlit dashboard.

Run from project root:
    streamlit run src/ui/dashboard.py
"""
from __future__ import annotations

import os
import sys
import json

# Ensure the project root is importable when launched via `streamlit run`.
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
from pyvis.network import Network

from src.database.db import get_driver
from src.models.predict import predict_associations, predict_exploits

# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Predictive CTI Graph Network",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Minimal dark-theme CSS ────────────────────────────────────────────────────

st.markdown("""
<style>
.metric-card {
    background: #1e1e2e; border: 1px solid #313244;
    border-radius: 8px; padding: 16px; text-align: center;
}
.metric-value { font-size: 2rem; font-weight: 700; color: #cba6f7; }
.metric-label { font-size: 0.8rem; color: #a6adc8; margin-top: 4px; }
.caveat-box {
    background: #2a1a0e; border-left: 4px solid #fab387;
    border-radius: 4px; padding: 10px 14px; margin: 8px 0;
    font-size: 0.85rem; color: #fab387;
}
.real-box {
    background: #0e2a1a; border-left: 4px solid #a6e3a1;
    border-radius: 4px; padding: 10px 14px; margin: 8px 0;
    font-size: 0.85rem; color: #a6e3a1;
}
</style>
""", unsafe_allow_html=True)


# ── Cached data fetchers ──────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def get_summary():
    with get_driver().session() as s:
        counts = {r["label"]: r["count"] for r in s.run(
            "MATCH (n) UNWIND labels(n) AS label "
            "RETURN label, count(*) AS count ORDER BY count DESC"
        )}
        rel_counts = {r["rel"]: r["count"] for r in s.run(
            "MATCH ()-[r]->() RETURN type(r) AS rel, count(*) AS count"
        )}
    return counts, rel_counts


@st.cache_data(ttl=300)
def get_all_actors():
    with get_driver().session() as s:
        return sorted([r["name"] for r in s.run(
            "MATCH (a:ThreatActor) RETURN a.name AS name ORDER BY a.name"
        )])


@st.cache_data(ttl=300)
def get_top_vendors(limit=10):
    with get_driver().session() as s:
        return [dict(r) for r in s.run("""
            MATCH (v:Vendor)<-[:AFFECTS]-(c:CVE)<-[:EXPLOITS]-(m:Malware)
            RETURN v.name AS vendor,
                   count(DISTINCT m) AS malware_count,
                   count(DISTINCT c) AS exploited_cves
            ORDER BY malware_count DESC LIMIT $limit
        """, limit=limit)]


@st.cache_data(ttl=300)
def get_actor_subgraph(actor_name: str):
    """Return nodes + edges for an actor's 2-hop neighbourhood."""
    with get_driver().session() as s:
        rows = list(s.run("""
            MATCH (a:ThreatActor {name: $name})-[:ASSOCIATED_WITH]->(m:Malware)
            OPTIONAL MATCH (m)-[:EXPLOITS]->(c:CVE)-[:AFFECTS]->(v:Vendor)
            RETURN a.name AS actor,
                   m.name AS malware, m.id AS malware_id,
                   c.id   AS cve,
                   v.name AS vendor
        """, name=actor_name))
    return [dict(r) for r in rows]


@st.cache_data(ttl=300)
def get_actor_info(actor_name: str):
    with get_driver().session() as s:
        r = s.run(
            "MATCH (a:ThreatActor {name: $n}) RETURN a", n=actor_name
        ).single()
    return dict(r["a"]) if r else {}


def models_available():
    return (
        os.path.exists(os.path.join("data", "gnn_ASSOCIATED_WITH.pt")),
        os.path.exists(os.path.join("data", "gnn_EXPLOITS.pt")),
    )


# ── Graph builder ─────────────────────────────────────────────────────────────

def build_pyvis(rows: list[dict], actor_name: str) -> str:
    net = Network(height="480px", width="100%", bgcolor="#1e1e2e",
                  font_color="#cdd6f4", directed=True, cdn_resources="remote")
    net.set_options(json.dumps({
        "physics": {"stabilization": {"iterations": 80}},
        "edges": {"arrows": {"to": {"enabled": True, "scaleFactor": 0.5}},
                  "color": {"opacity": 0.7}, "smooth": False},
        "interaction": {"hover": True},
    }))

    added = set()

    def node(nid, label, color, shape="dot", size=18):
        if nid not in added:
            net.add_node(nid, label=label, color=color,
                         shape=shape, size=size, title=label)
            added.add(nid)

    node(actor_name, actor_name, "#cba6f7", shape="star", size=28)

    for r in rows:
        if r["malware"]:
            mid = r["malware_id"] or r["malware"]
            node(mid, r["malware"], "#89b4fa")
            if (actor_name, mid) not in added:
                net.add_edge(actor_name, mid, color="#89b4fa", title="ASSOCIATED_WITH")
                added.add((actor_name, mid))
        if r["cve"]:
            node(r["cve"], r["cve"], "#f38ba8", shape="diamond", size=14)
            mid = r["malware_id"] or r["malware"]
            if (mid, r["cve"]) not in added:
                net.add_edge(mid, r["cve"], color="#f38ba8",
                             title="EXPLOITS", dashes=True)
                added.add((mid, r["cve"]))
        if r["vendor"]:
            node(r["vendor"], r["vendor"], "#a6e3a1", shape="square", size=14)
            if (r["cve"], r["vendor"]) not in added:
                net.add_edge(r["cve"], r["vendor"], color="#a6e3a1", title="AFFECTS")
                added.add((r["cve"], r["vendor"]))

    # Generate self-contained HTML in memory (no temp files — avoids the
    # Windows file-lock error and renders correctly inside Streamlit).
    return net.generate_html(notebook=False)


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🛡️ CTI Graph Network")
    st.markdown("---")
    st.markdown("**Select Threat Actor**")
    actors = get_all_actors()
    selected = st.selectbox("", actors, index=actors.index("APT29") if "APT29" in actors else 0)
    top_k = st.slider("Prediction top-K", 5, 20, 10)
    st.markdown("---")
    aw_avail, ex_avail = models_available()
    st.markdown(f"**Models**")
    st.markdown(f"{'✅' if aw_avail else '❌'} ASSOCIATED_WITH (real)")
    st.markdown(f"{'✅' if ex_avail else '❌'} EXPLOITS (experimental)")
    st.markdown("---")
    st.caption("Data: MITRE ATT&CK + CISA KEV")


# ── Header ────────────────────────────────────────────────────────────────────

st.markdown("# 🛡️ Predictive CTI Graph Network")
st.markdown(f"### Threat Actor: **{selected}**")

info = get_actor_info(selected)
if info.get("aliases"):
    aliases = info["aliases"]
    if isinstance(aliases, list):
        aliases = ", ".join(aliases)
    st.caption(f"Also known as: {aliases}")

# ── Summary metrics ───────────────────────────────────────────────────────────

counts, rel_counts = get_summary()
c1, c2, c3, c4, c5 = st.columns(5)
metrics = [
    (c1, "Threat Actors", counts.get("ThreatActor", 0)),
    (c2, "Malware / Tools", counts.get("Malware", 0)),
    (c3, "CVEs (KEV)", counts.get("CVE", 0)),
    (c4, "Vendors", counts.get("Vendor", 0)),
    (c5, "Relationships", sum(rel_counts.values())),
]
for col, label, val in metrics:
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{val:,}</div>
            <div class="metric-label">{label}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs(
    ["🔮 Predictions", "🕸️ Subgraph", "📊 Analytics", "ℹ️ About"]
)


# ── Tab 1: Predictions ────────────────────────────────────────────────────────

with tab1:
    col_real, col_exp = st.columns(2)

    with col_real:
        st.markdown("#### 🟢 Real Model — Malware Associations")
        st.markdown(
            '<div class="real-box">Trained on 1,145 real MITRE edges. '
            'ROC-AUC 0.68 on held-out test set. '
            'Predicts: which malware/tools is this actor likely to also use?</div>',
            unsafe_allow_html=True,
        )
        if aw_avail:
            with st.spinner("Running predictions..."):
                out = predict_associations(selected, top_k=top_k)
            if "error" in out:
                st.warning(out["error"])
            else:
                df = pd.DataFrame(out["results"])
                df["probability"] = df["probability"].apply(lambda x: f"{x:.4f}")
                st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("Run `python -m src.models.train_all` first.")

    with col_exp:
        st.markdown("#### 🟠 Experimental Model — CVE Exploit Predictions")
        st.markdown(
            '<div class="caveat-box">⚠️ Trained on heuristic edges — '
            'exploratory, not ground truth. '
            'ROC-AUC 0.99 reflects heuristic re-learning, not real discovery.</div>',
            unsafe_allow_html=True,
        )
        if ex_avail:
            with st.spinner("Running predictions..."):
                out = predict_exploits(selected, top_k=top_k)
            if "error" in out:
                st.warning(out["error"])
            else:
                df = pd.DataFrame(out["results"])
                df["probability"] = df["probability"].apply(lambda x: f"{x:.4f}")
                st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("Run `python -m src.models.train_all` first.")


# ── Tab 2: Subgraph ───────────────────────────────────────────────────────────

with tab2:
    st.markdown(f"#### Attack graph for **{selected}**")
    st.caption("★ Actor  •  Blue = Malware  •  Red ◆ = CVE  •  Green ■ = Vendor")
    rows = get_actor_subgraph(selected)
    if rows:
        html = build_pyvis(rows, selected)
        components.html(html, height=500, scrolling=False)
        malware_list = list({r["malware"] for r in rows if r["malware"]})
        st.caption(f"Known malware: {', '.join(sorted(malware_list))}")
    else:
        st.info("No subgraph data found for this actor.")


# ── Tab 3: Analytics ─────────────────────────────────────────────────────────

with tab3:
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### Most-targeted vendors")
        vendors = get_top_vendors()
        if vendors:
            df = pd.DataFrame(vendors)
            st.bar_chart(df.set_index("vendor")["malware_count"], use_container_width=True)
            st.dataframe(df, use_container_width=True, hide_index=True)

    with col_b:
        st.markdown("#### Relationship breakdown")
        if rel_counts:
            df_r = pd.DataFrame(
                [{"relationship": k, "count": v} for k, v in rel_counts.items()]
            ).sort_values("count", ascending=False)
            st.bar_chart(df_r.set_index("relationship")["count"], use_container_width=True)
            st.dataframe(df_r, use_container_width=True, hide_index=True)


# ── Tab 4: About ──────────────────────────────────────────────────────────────

with tab4:
    st.markdown("""
#### Project overview
This dashboard explores cyber threat intelligence as a knowledge graph and uses
Graph Neural Networks (GNNs) to predict likely threat actor–malware associations.

**Data sources** (real, free, no API keys)
- **MITRE ATT&CK** — 174 threat actors, 821 malware/tools, 1,145 real ASSOCIATED_WITH edges
- **CISA Known Exploited Vulnerabilities** — all confirmed-exploited CVEs with vendor and CWE mappings

**Machine learning**
- Architecture: 2-layer GCN encoder + MLP link-prediction decoder
- Target 1 (headline): ASSOCIATED_WITH — **ROC-AUC 0.68, AP 0.74** on real held-out edges
- Target 2 (experimental): EXPLOITS — heuristic-trained; high score reflects label quality, not discovery

**Honest caveats**
- There is no publicly available, structured, free dataset mapping malware to CVEs at scale.
  The EXPLOITS edges are derived from heuristics and should be treated as exploratory.
- The 0.68 ASSOCIATED_WITH ROC-AUC is the scientifically defensible result.

**Stack**
Neo4j 5.26 · PyTorch Geometric · sentence-transformers (all-MiniLM-L6-v2) · Streamlit · Pyvis
    """)
