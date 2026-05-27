"""Phase 5 — Inference for both trained models.

predict_associations(actor)  -> real model (ASSOCIATED_WITH):
    which OTHER malware/tools is this actor likely to also use?

predict_exploits(actor)      -> experimental model (EXPLOITS):
    which CVEs might this actor's malware exploit? (heuristic-trained — caveat)

Run from project root:
    python -m src.models.predict "APT29"
"""
from __future__ import annotations

import os
import sys

import torch

from src.models.gnn import CTILinkPredictor

DATA_PATH = os.path.join("data", "graph_data.pt")


def _load_graph():
    return torch.load(DATA_PATH, weights_only=False)


def _load_model(target):
    path = os.path.join("data", f"gnn_{target}.pt")
    if not os.path.exists(path):
        return None
    ckpt = torch.load(path, weights_only=False)
    model = CTILinkPredictor(ckpt["in_channels"], ckpt["hidden"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def _encode(data, model):
    # Message-passing graph = all edges undirected (inference uses full graph).
    e = data["edge_index_all"]
    mp = torch.cat([e, e.flip(0)], dim=1)
    return model.encode(data["x"], mp)


def _actor_indices(meta, actor_substring):
    return [i for i, m in enumerate(meta)
            if m["label"] == "ThreatActor"
            and actor_substring.lower() in str(m["name"]).lower()]


def _actor_malware(data, actor_ids):
    meta = data["node_meta"]
    malware = {i for i, m in enumerate(meta) if m["label"] == "Malware"}
    e = data["edge_index_all"]
    actor_set = set(actor_ids)
    found = set()
    for s, d in zip(e[0].tolist(), e[1].tolist()):
        if s in actor_set and d in malware:
            found.add(d)
    return found


@torch.no_grad()
def predict_associations(actor_substring, top_k=10):
    data = _load_graph()
    model = _load_model("ASSOCIATED_WITH")
    if model is None:
        return {"error": "ASSOCIATED_WITH model not trained yet."}
    meta = data["node_meta"]
    actor_ids = _actor_indices(meta, actor_substring)
    if not actor_ids:
        return {"error": f"No actor matching '{actor_substring}'."}
    z = _encode(data, model)

    known = _actor_malware(data, actor_ids)
    malware_idx = [i for i, m in enumerate(meta) if m["label"] == "Malware"]
    candidates = [i for i in malware_idx if i not in known]
    cand_t = torch.tensor(candidates)

    a = actor_ids[0]
    src = torch.full((len(candidates),), a, dtype=torch.long)
    probs = torch.sigmoid(model.decode(z, torch.stack([src, cand_t])))
    order = torch.argsort(probs, descending=True)[:top_k]
    return {"results": [
        {"rank": r + 1, "malware": meta[candidates[i]]["name"],
         "id": meta[candidates[i]]["id"], "probability": round(probs[i].item(), 4)}
        for r, i in enumerate(order.tolist())
    ]}


@torch.no_grad()
def predict_exploits(actor_substring, top_k=10):
    data = _load_graph()
    model = _load_model("EXPLOITS")
    if model is None:
        return {"error": "EXPLOITS model not trained yet."}
    meta = data["node_meta"]
    actor_ids = _actor_indices(meta, actor_substring)
    if not actor_ids:
        return {"error": f"No actor matching '{actor_substring}'."}
    actor_malware = _actor_malware(data, actor_ids)
    if not actor_malware:
        return {"error": f"'{actor_substring}' has no linked malware."}
    z = _encode(data, model)

    cve_idx = [i for i, m in enumerate(meta) if m["label"] == "CVE"]
    cve_t = torch.tensor(cve_idx)
    best = torch.zeros(len(cve_idx))
    for mi in actor_malware:
        src = torch.full((len(cve_idx),), mi, dtype=torch.long)
        probs = torch.sigmoid(model.decode(z, torch.stack([src, cve_t])))
        best = torch.maximum(best, probs)
    order = torch.argsort(best, descending=True)[:top_k]
    return {"results": [
        {"rank": r + 1, "cve": meta[cve_idx[i]]["id"],
         "probability": round(best[i].item(), 4)}
        for r, i in enumerate(order.tolist())
    ], "caveat": "Trained on heuristic edges — exploratory, not ground truth."}


if __name__ == "__main__":
    actor = sys.argv[1] if len(sys.argv) > 1 else "APT29"

    print(f"\n=== REAL MODEL: malware {actor} is likely to also use ===")
    out = predict_associations(actor)
    if "error" in out:
        print(" ", out["error"])
    else:
        for r in out["results"]:
            print(f"  {r['rank']:2d}. {r['malware']:28s} ({r['id']:7s})  p={r['probability']:.4f}")

    print(f"\n=== EXPERIMENTAL MODEL: CVEs {actor} might exploit ===")
    out = predict_exploits(actor)
    if "error" in out:
        print(" ", out["error"])
    else:
        print(f"  [{out['caveat']}]")
        for r in out["results"]:
            print(f"  {r['rank']:2d}. {r['cve']:18s}  p={r['probability']:.4f}")
