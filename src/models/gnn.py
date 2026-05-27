"""Phase 5 — GNN model: a GCN encoder + MLP link-prediction decoder.

Adapted from the project blueprint. The encoder learns node embeddings via
two-hop neighbourhood aggregation; the decoder scores whether a (malware, CVE)
pair should be connected by an EXPLOITS edge.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv


class CTILinkPredictor(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 128):
        super().__init__()
        # Two GCN layers: 1-hop then 2-hop structural context.
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)

        # MLP that scores a pair of node embeddings -> link probability.
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(hidden_channels, 1),
        )

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

    def decode(self, z, edge_label_index):
        src = z[edge_label_index[0]]
        dst = z[edge_label_index[1]]
        edge_features = torch.cat([src, dst], dim=-1)
        # Return raw logits; we use BCEWithLogitsLoss for numerical stability.
        return self.classifier(edge_features).squeeze(-1)

    def forward(self, x, edge_index, edge_label_index):
        z = self.encode(x, edge_index)
        return self.decode(z, edge_label_index)
