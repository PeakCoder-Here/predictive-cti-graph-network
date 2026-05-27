"""Phase 5 — Train a link-prediction GNN for a chosen relationship type.

The target relation's edges are held out and predicted; all other relations
(plus the training split of the target) form the message-passing graph.

Targets:
  ASSOCIATED_WITH  ThreatActor -> Malware   (1,145 REAL edges — headline model)
  EXPLOITS         Malware     -> CVE        (heuristic edges — experimental)

Run from project root:
    python -m src.models.train --target ASSOCIATED_WITH
    python -m src.models.train --target EXPLOITS --sources all
"""
from __future__ import annotations

import argparse
import os

import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from src.models.gnn import CTILinkPredictor

DATA_PATH = os.path.join("data", "graph_data.pt")

# Which node labels sit at each end of a target relation (src_label, dst_label).
REL_ENDPOINTS = {
    "ASSOCIATED_WITH": ("ThreatActor", "Malware"),
    "EXPLOITS": ("Malware", "CVE"),
    "AFFECTS": ("CVE", "Vendor"),
}


def _model_path(target: str) -> str:
    return os.path.join("data", f"gnn_{target}.pt")


def _load(target: str, sources: str):
    data = torch.load(DATA_PATH, weights_only=False)
    meta = data["node_meta"]
    rels = data["edge_types"]
    srcs = data["edge_sources"]
    all_edges = data["edge_index_all"]

    target_mask, struct_s, struct_d = [], [], []
    for i in range(all_edges.size(1)):
        s, d = all_edges[0, i].item(), all_edges[1, i].item()
        if rels[i] == target:
            if sources == "all" or srcs[i] == sources:
                target_mask.append(i)
        else:
            struct_s += [s, d]      # undirected for message passing
            struct_d += [d, s]

    target_edges = all_edges[:, target_mask]
    structural = torch.tensor([struct_s, struct_d], dtype=torch.long)

    src_label, dst_label = REL_ENDPOINTS[target]
    src_nodes = [i for i, m in enumerate(meta) if m["label"] == src_label]
    dst_nodes = [i for i, m in enumerate(meta) if m["label"] == dst_label]
    return data, structural, target_edges, src_nodes, dst_nodes


def _split(pos, train=0.7, val=0.15, seed=42):
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(pos.size(1), generator=g)
    n = pos.size(1)
    n_tr, n_va = int(train * n), int(val * n)
    return pos[:, perm[:n_tr]], pos[:, perm[n_tr:n_tr + n_va]], pos[:, perm[n_tr + n_va:]]


def _sample_negatives(num, src_nodes, dst_nodes, positives_set, gen):
    s_t, d_t = torch.tensor(src_nodes), torch.tensor(dst_nodes)
    neg_s, neg_d = [], []
    guard = 0
    while len(neg_s) < num and guard < 1000:
        guard += 1
        s = s_t[torch.randint(len(s_t), (num,), generator=gen)]
        d = d_t[torch.randint(len(d_t), (num,), generator=gen)]
        for si, di in zip(s.tolist(), d.tolist()):
            if (si, di) not in positives_set:
                neg_s.append(si); neg_d.append(di)
                if len(neg_s) >= num:
                    break
    return torch.tensor([neg_s, neg_d], dtype=torch.long)


@torch.no_grad()
def _evaluate(model, x, mp_edges, pos, neg):
    model.eval()
    eli = torch.cat([pos, neg], dim=1)
    y = torch.cat([torch.ones(pos.size(1)), torch.zeros(neg.size(1))])
    scores = torch.sigmoid(model(x, mp_edges, eli)).cpu().numpy()
    return roc_auc_score(y, scores), average_precision_score(y, scores)


def train(target="ASSOCIATED_WITH", sources="all", epochs=200, hidden=128, lr=0.01, seed=42):
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)

    data, structural, pos, src_nodes, dst_nodes = _load(target, sources)
    print(f"Target: {target} ({REL_ENDPOINTS[target][0]} -> {REL_ENDPOINTS[target][1]})")
    print(f"Positive edges: {pos.size(1)} | source filter: {sources}")
    if pos.size(1) < 20:
        print("WARNING: too few positives for a meaningful split.")
        return

    x = data["x"]
    train_pos, val_pos, test_pos = _split(pos, seed=seed)
    train_undir = torch.cat([train_pos, train_pos.flip(0)], dim=1)
    mp_edges = torch.cat([structural, train_undir], dim=1)

    positives_set = {(s, d) for s, d in zip(pos[0].tolist(), pos[1].tolist())}
    val_neg = _sample_negatives(val_pos.size(1), src_nodes, dst_nodes, positives_set, gen)
    test_neg = _sample_negatives(test_pos.size(1), src_nodes, dst_nodes, positives_set, gen)

    model = CTILinkPredictor(in_channels=x.size(1), hidden_channels=hidden)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        train_neg = _sample_negatives(train_pos.size(1), src_nodes, dst_nodes, positives_set, gen)
        eli = torch.cat([train_pos, train_neg], dim=1)
        y = torch.cat([torch.ones(train_pos.size(1)), torch.zeros(train_neg.size(1))])
        loss = loss_fn(model(x, mp_edges, eli), y)
        loss.backward()
        opt.step()
        if epoch % 20 == 0 or epoch == 1:
            auc, ap = _evaluate(model, x, mp_edges, val_pos, val_neg)
            print(f"  epoch {epoch:3d} | loss {loss.item():.4f} | val ROC-AUC {auc:.4f} | val AP {ap:.4f}")

    test_auc, test_ap = _evaluate(model, x, mp_edges, test_pos, test_neg)
    print(f"\nTEST  ROC-AUC {test_auc:.4f} | Average Precision {test_ap:.4f}")

    torch.save({
        "state_dict": model.state_dict(),
        "in_channels": x.size(1),
        "hidden": hidden,
        "target": target,
        "sources": sources,
    }, _model_path(target))
    print(f"Saved model to {_model_path(target)}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--target", default="ASSOCIATED_WITH", choices=list(REL_ENDPOINTS))
    p.add_argument("--sources", default="all")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--hidden", type=int, default=128)
    args = p.parse_args()
    train(target=args.target, sources=args.sources, epochs=args.epochs, hidden=args.hidden)
