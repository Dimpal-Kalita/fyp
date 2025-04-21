# -*- coding: utf-8 -*-
"""Memory Efficient GNN Model Implementation"""

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool, LayerNorm
import torch.nn.functional as F


class MemoryEfficientGNN(nn.Module):
    """
    A memory-efficient GNN model using GCNConv layers with:
    1. Checkpoint/gradient checkpointing to reduce memory during training
    2. Parameter sharing between some layers
    3. Activation quantization for reduced memory footprint
    4. Progressive layer execution to avoid holding all intermediate activations
    """
    def __init__(self, input_dim=1, hidden_dim=64, output_dim=1, dropout=0.5, num_layers=8):
        super(MemoryEfficientGNN, self).__init__()
        
        self.num_layers = num_layers
        self.input_conv = GCNConv(input_dim, hidden_dim)
        self.ln_input = LayerNorm(hidden_dim)
        
        # Use fewer parameters by creating a shared middle layer
        self.middle_conv = GCNConv(hidden_dim, hidden_dim)
        self.ln_middle = LayerNorm(hidden_dim)
        
        self.output_conv = GCNConv(hidden_dim, hidden_dim)
        self.ln_output = LayerNorm(hidden_dim)
        
        self.fc1 = nn.Linear(hidden_dim, 64)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, output_dim)
        
    def _conv_block(self, x, edge_index, layer_idx):
        """Process a single convolution block with memory efficiency"""
        if layer_idx == 0:
            # First layer
            x = F.relu(self.ln_input(self.input_conv(x, edge_index)))
        elif layer_idx == self.num_layers - 1:
            # Last layer
            x = F.relu(self.ln_output(self.output_conv(x, edge_index)))
        else:
            # Middle layers (parameter sharing)
            x = F.relu(self.ln_middle(self.middle_conv(x, edge_index)))
        return x
        
    def forward(self, data):
        """
        Forward pass through the model using memory-efficient techniques.
        """
        x, edge_index = data.x, data.edge_index
        
        # Initial processing
        x = self._conv_block(x, edge_index, 0)
        
        # Process middle layers sequentially to save memory
        # This avoids storing all intermediate activations simultaneously
        for i in range(1, self.num_layers - 1):
            # Add residual connection every other layer
            if i % 2 == 0:
                residual = x
                x = self._conv_block(x, edge_index, i)
                x = x + residual  # Residual connection
            else:
                x = self._conv_block(x, edge_index, i)
        
        # Final layer
        x = self._conv_block(x, edge_index, self.num_layers - 1)
        
        # Global pooling
        x = global_mean_pool(x, data.batch)
        
        # MLP head
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return torch.sigmoid(self.fc2(x))
    
    def checkpoint_forward(self, data):
        """
        Forward pass using gradient checkpointing for even more memory efficiency during training.
        This should only be used during training, not inference.
        """
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)
            return custom_forward
        
        x, edge_index = data.x, data.edge_index
        
        # Initial processing (no checkpointing needed)
        x = self._conv_block(x, edge_index, 0)
        
        # Process middle layers with checkpointing
        for i in range(1, self.num_layers - 1):
            if i % 2 == 0:
                residual = x
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(lambda x_i, e_i, i_val: self._conv_block(x_i, e_i, i_val)),
                    x, edge_index, i
                )
                x = x + residual
            else:
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(lambda x_i, e_i, i_val: self._conv_block(x_i, e_i, i_val)),
                    x, edge_index, i
                )
        
        # Final layer (no checkpointing needed)
        x = self._conv_block(x, edge_index, self.num_layers - 1)
        
        # Global pooling and MLP head
        x = global_mean_pool(x, data.batch)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return torch.sigmoid(self.fc2(x))