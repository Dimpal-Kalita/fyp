# -*- coding: utf-8 -*-
"""Memory Efficient GNN Model Implementation (Improved)"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch_geometric.nn import GCNConv, global_mean_pool, LayerNorm


class MemoryEfficientGNN(nn.Module):
    """
    Memory-Efficient GNN using:
    - Parameter sharing
    - Optional gradient checkpointing
    - Optional residual connections
    - LayerNorm and dropout for regularization
    """

    def __init__(self, input_dim=12, hidden_dim=64, output_dim=1, dropout=0.5, num_layers=8,
                 use_checkpointing=False, use_residual=True):
        super(MemoryEfficientGNN, self).__init__()

        self.num_layers = num_layers
        self.use_checkpointing = use_checkpointing
        self.use_residual = use_residual

        self.input_conv = GCNConv(input_dim, hidden_dim)
        self.ln_input = LayerNorm(hidden_dim)

        # Shared middle layer
        self.middle_conv = GCNConv(hidden_dim, hidden_dim)
        self.ln_middle = LayerNorm(hidden_dim)

        self.output_conv = GCNConv(hidden_dim, hidden_dim)
        self.ln_output = LayerNorm(hidden_dim)

        self.fc1 = nn.Linear(hidden_dim, 64)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, output_dim)

    def _conv_block(self, x, edge_index, layer_idx):
        if layer_idx == 0:
            x = F.relu(self.ln_input(self.input_conv(x, edge_index)))
        elif layer_idx == self.num_layers - 1:
            x = F.relu(self.ln_output(self.output_conv(x, edge_index)))
        else:
            x = F.relu(self.ln_middle(self.middle_conv(x, edge_index)))
        return x

    def _maybe_checkpoint(self, func, *args):
        """Helper to apply checkpointing only if enabled"""
        if self.use_checkpointing and self.training:
            return checkpoint(func, *args)
        else:
            return func(*args)

    def forward(self, data):
        """
        Forward pass with optional memory-efficient checkpointing and residuals.
        """
        x, edge_index = data.x, data.edge_index

        # First layer
        x = self._conv_block(x, edge_index, 0)

        # Middle layers
        for i in range(1, self.num_layers - 1):
            layer_func = lambda x_i, ei_i, li_i: self._conv_block(x_i, ei_i, li_i)
            if self.use_residual and i % 2 == 0:
                residual = x
                x = self._maybe_checkpoint(layer_func, x, edge_index, i)
                x = x + residual
            else:
                x = self._maybe_checkpoint(layer_func, x, edge_index, i)

        # Final layer
        x = self._conv_block(x, edge_index, self.num_layers - 1)

        # Global pooling
        x = global_mean_pool(x, data.batch)

        # MLP Head
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)
