# -*- coding: utf-8 -*-
"""Model Training and Evaluation Module"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool, LayerNorm
from sklearn.metrics import roc_curve, precision_recall_curve
from data_processing import load_data_split, set_seeds
from plots import plot_metrics_and_curves
# from gnn_model import GNNModel
from memory_efficient_gnn import MemoryEfficientGNN


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs):
    """
    Trains the GNN model and evaluates it on validation data.
    """
    train_losses, val_losses, train_acc, val_acc, train_rmsd, val_rmsd = [], [], [], [], [], []

    for epoch in range(epochs):
        model.train()
        total_loss, correct, total, rmsd = 0., 0., 0., 0.

        for data in train_loader:
            optimizer.zero_grad()
            output = model(data).squeeze()
            # Ensure both tensors have the same shape
            if output.dim() == 0 and data.y.dim() == 1:
                output = output.unsqueeze(0)
            loss = criterion(output, data.y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * data.num_graphs
            correct += ((output > 0.5) == data.y).sum().item()
            total += data.num_graphs
            rmsd += torch.sum((output - data.y)**2)

        train_loss = total_loss / len(train_loader.dataset)
        train_accuracy = correct / total
        rmsd_total = torch.sqrt(rmsd/len(train_loader.dataset))
        train_losses.append(train_loss)
        train_acc.append(train_accuracy)
        train_rmsd.append(rmsd_total)

        val_loss, val_accuracy, val_rmsd_value = evaluate_model(model, val_loader, criterion)
        val_losses.append(val_loss)
        val_acc.append(val_accuracy)
        val_rmsd.append(val_rmsd_value)

        scheduler.step(val_loss)

        print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, Train RMSD: {rmsd_total:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}, Val RMSD: {val_rmsd_value:.4f}")

    return train_losses, val_losses, train_acc, val_acc, train_rmsd, val_rmsd

def evaluate_model(model, loader, criterion):
    """
    Evaluates the model on the given dataset.
    """
    model.eval()
    total_loss, correct, total = 0, 0, 0
    rmsd = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for data in loader:
            output = model(data).squeeze()
            # Ensure both tensors have the same shape
            if output.dim() == 0 and data.y.dim() == 1:
                output = output.unsqueeze(0)
            loss = criterion(output, data.y)
            total_loss += loss.item() * data.num_graphs
            correct += ((output > 0.5) == data.y).sum().item()
            total += data.num_graphs
            rmsd += torch.sum((output - data.y)**2)

            all_preds.extend(output.cpu().numpy())
            all_labels.extend(data.y.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    accuracy = correct / total
    rmsd_total = torch.sqrt(rmsd/len(loader.dataset))
    
    return avg_loss, accuracy, rmsd_total

if __name__ == "__main__":
    set_seeds(42)
    data_dir = "./GCN/data"  # Path to PDB files
    print("Loading data...")
    train_data, val_data, test_data = load_data_split(data_dir)
    print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32)
    test_loader = DataLoader(test_data, batch_size=32)
    print("Data loaders created.")
    model = MemoryEfficientGNN()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0009, weight_decay=0.003)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min')

    print("Training the model...")
    train_losses, val_losses, train_acc, val_acc, train_rmsd, val_rmsd = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler, epochs=20
    )

    print("Evaluating the model...")    
    #prediction.py called

    test_loss, test_acc, test_rmsd = evaluate_model(model, test_loader, criterion)
    
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}, Test RMSD: {test_rmsd:.4f}")

    # We need to collect predictions and labels for plotting
    all_preds, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for data in test_loader:
            output = model(data).squeeze()
            # Ensure both tensors have the same shape
            if output.dim() == 0 and data.y.dim() == 1:
                output = output.unsqueeze(0)
            all_preds.extend(output.cpu().numpy())
            all_labels.extend(data.y.cpu().numpy())
    
    # Now we can call the plotting function with both arguments
    plot_metrics_and_curves(train_losses, val_losses, train_acc, val_acc, all_preds, all_labels)