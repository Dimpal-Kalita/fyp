# -*- coding: utf-8 -*-
"""GNN Model Definition"""

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool, LayerNorm

class GNNModel(nn.Module):
    """
    A simple GNN model using GCNConv layers with residual connections and LayerNorm.
    """
    def __init__(self, input_dim=1, hidden_dim=64, output_dim=1, dropout=0.5):
        super(GNNModel, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.ln1 = LayerNorm(hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.ln2 = LayerNorm(hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.ln3 = LayerNorm(hidden_dim)
        self.conv4 = GCNConv(hidden_dim, hidden_dim)
        self.ln4 = LayerNorm(hidden_dim)
        self.conv5 = GCNConv(hidden_dim, hidden_dim)
        self.ln5 = LayerNorm(hidden_dim)
        self.conv6 = GCNConv(hidden_dim, hidden_dim)
        self.ln6 = LayerNorm(hidden_dim)
        self.conv7 = GCNConv(hidden_dim, hidden_dim)
        self.ln7 = LayerNorm(hidden_dim)
        self.conv8 = GCNConv(hidden_dim, hidden_dim)
        self.ln8 = LayerNorm(hidden_dim)
        
        self.fc1 = nn.Linear(hidden_dim, 64)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, output_dim)
    
    def forward(self, data):
        """
        Forward pass through the model.
        """
        x, edge_index = data.x, data.edge_index
        x1 = torch.relu(self.conv1(x, edge_index))
        x2 = torch.relu(self.ln2(self.conv2(x1, edge_index))) + x1
        x3 = torch.relu(self.ln3(self.conv3(x2, edge_index))) + x2
        x4 = torch.relu(self.ln4(self.conv4(x3, edge_index))) + x3
        x5 = torch.relu(self.ln5(self.conv5(x4, edge_index))) + x4
        x6 = torch.relu(self.ln6(self.conv6(x5, edge_index))) + x5
        x7 = torch.relu(self.ln7(self.conv7(x6, edge_index))) + x6
        x8 = torch.relu(self.ln8(self.conv8(x7, edge_index))) + x7 # Residual connection
        x = global_mean_pool(x8, data.batch)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        return torch.sigmoid(self.fc2(x))