"""
GNN models for quark/gluon jet classification.

GCNClassifier: Simple GCN baseline
ParticleNet: DynamicEdgeConv-based model (adapted from ParticleNet paper)
"""

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool, DynamicEdgeConv


class GCNClassifier(nn.Module):
    """Simple 3-layer GCN + global mean pooling + FC classifier."""

    def __init__(self, in_channels=3, hidden=128, num_classes=2):
        super().__init__()
        self.input_bn = nn.BatchNorm1d(in_channels)
        self.conv1 = GCNConv(in_channels, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.conv3 = GCNConv(hidden, hidden)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.bn2 = nn.BatchNorm1d(hidden)
        self.bn3 = nn.BatchNorm1d(hidden)
        self.classifier = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.input_bn(x)

        x = self.bn1(torch.relu(self.conv1(x, edge_index)))
        x = self.bn2(torch.relu(self.conv2(x, edge_index)))
        x = self.bn3(torch.relu(self.conv3(x, edge_index)))

        x = global_mean_pool(x, batch)
        return self.classifier(x)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def _make_mlp(channels):
    """Build MLP with BN and ReLU for EdgeConv."""
    layers = []
    for i in range(len(channels) - 1):
        layers.append(nn.Linear(channels[i], channels[i + 1]))
        layers.append(nn.BatchNorm1d(channels[i + 1]))
        layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class ParticleNet(nn.Module):
    """ParticleNet-style classifier using DynamicEdgeConv.

    3 EdgeConv blocks with dynamic kNN in feature space,
    followed by global mean pooling and FC classifier.
    """

    def __init__(self, in_channels=3, k=16, num_classes=2):
        super().__init__()
        self.k = k
        self.input_bn = nn.BatchNorm1d(in_channels)

        # EdgeConv blocks: input channels are doubled (x_i concat x_j - x_i)
        self.conv1 = DynamicEdgeConv(_make_mlp([2 * in_channels, 64, 64, 64]), k=k, aggr="max")
        self.conv2 = DynamicEdgeConv(_make_mlp([2 * 64, 128, 128, 128]), k=k, aggr="max")
        self.conv3 = DynamicEdgeConv(_make_mlp([2 * 128, 256, 256, 256]), k=k, aggr="max")

        # Shortcut projections
        self.shortcut1 = nn.Linear(in_channels, 64)
        self.shortcut2 = nn.Linear(64, 128)
        self.shortcut3 = nn.Linear(128, 256)

        self.classifier = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes),
        )

    def forward(self, data):
        x, batch = data.x, data.batch
        x = self.input_bn(x)

        # Block 1
        x1 = self.conv1(x, batch) + self.shortcut1(x)

        # Block 2
        x2 = self.conv2(x1, batch) + self.shortcut2(x1)

        # Block 3
        x3 = self.conv3(x2, batch) + self.shortcut3(x2)

        # Global pooling + classifier
        out = global_mean_pool(x3, batch)
        return self.classifier(out)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
