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
from data_processing import load_data_splits, MolecularFeatureExtractor, ProteinLigandDataset
import pandas as pd
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs):
    """
    Trains the GNN model and evaluates it on validation data.
    """
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0., 0, 0

        for data in train_loader:
            optimizer.zero_grad()
            output = model(data).view(-1)
            y = data.y.float().view(-1)
            assert output.shape == y.shape, f"Shape mismatch: output {output.shape}, y {y.shape}"
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * data.num_graphs
            probs = torch.sigmoid(output)
            preds = (probs > 0.5).float()
            correct += (preds == y.float()).sum().item()
            total += data.num_graphs

        train_loss = total_loss / len(train_loader.dataset)
        train_acc = correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        val_loss, val_acc = evaluate_model(model, val_loader, criterion)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        scheduler.step(val_loss)

        print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    return train_losses, val_losses, train_accs, val_accs

def evaluate_model(model, loader, criterion):
    """
    Evaluates the model on the given dataset.
    """
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for data in loader:
            output = model(data).view(-1)
            y = data.y.float().view(-1)
            assert output.shape == y.shape, f"Shape mismatch: output {output.shape}, y {y.shape}"
            loss = criterion(output, y)
            total_loss += loss.item() * data.num_graphs
            probs = torch.sigmoid(output)
            preds = (probs > 0.5).float()
            correct += (preds == y.float()).sum().item()
            total += data.num_graphs
            all_preds.extend(probs.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    acc = correct / total
    return avg_loss, acc

def store_preprocessed_data(train_data, csv_path):
    """
    Stores preprocessed data in a CSV file.
    """
    # Convert train_data to a list of dictionaries
    data_list = []
    for data in train_data:
        print(data)
        data_list.append(data)
    # Create a DataFrame from the list of dictionaries
    df = pd.DataFrame(data_list)

    # Save the DataFrame to a CSV file
    df.to_csv(csv_path, index=False)
    print(f"Preprocessed data saved to {csv_path}")


if __name__ == "__main__":
    set_seeds(42)
    data_dir = "./GCN/data"  # Path to PDB files
    print("Loading data...")
    train_data, val_data, test_data = load_data_split(data_dir)    
    print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    store_preprocessed_data(train_data,"./output/processed_train_data.csv")

    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32)
    test_loader = DataLoader(test_data, batch_size=32)
    print("Data loaders created.")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MemoryEfficientGNN(
            input_dim=12,
            hidden_dim=64,
            output_dim=1,
            dropout=0.5,
            num_layers=8,
            use_checkpointing=True,
            use_residual=True
    ).to(device)

    # Compute class weights for binary classification
    # Collect all training labels (flattened)
    y_train = torch.cat([data.y.view(-1) for data in train_data]).cpu().numpy()
    classes = np.unique(y_train)
    class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    # For BCEWithLogitsLoss, pos_weight is the ratio of negative to positive samples
    # pos_weight = weight for positive class (1)
    if len(class_weights) == 2:
        pos_weight = torch.tensor(class_weights[0] / class_weights[1], dtype=torch.float).to(device)
    else:
        pos_weight = torch.tensor(1.0, dtype=torch.float).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = optim.Adam(model.parameters(), lr=0.0009, weight_decay=0.003)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min')

    print("Training the model...")
    train_losses, val_losses, train_accs, val_accs = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler, epochs=20
    )

    # Report final train and val accuracy
    print(f"Final Train Accuracy: {train_accs[-1]:.4f}")
    print(f"Final Val Accuracy: {val_accs[-1]:.4f}")

    # Save the trained model
    model_save_path = "./output/trained_gnn_model.pt"
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")
    
    print("Evaluating the model...")    
    #prediction.py called

    test_loss, test_acc = evaluate_model(model, test_loader, criterion)
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

    # We need to collect predictions and labels for plotting
    all_preds, all_labels = [], []
    all_true_energies, all_pred_energies = [], []
    model.eval()
    with torch.no_grad():
        for data in test_loader:
            output = model(data).view(-1)
            y = data.y.float().view(-1)
            assert output.shape == y.shape, f"Shape mismatch: output {output.shape}, y {y.shape}"
            probs = torch.sigmoid(output)
            all_preds.extend(probs.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            # For regression plot: get actual binding energy (before binarization)
            # Here, y is already normalized/binarized, so we need to parse the original binding energy
            # If you have access to the original binding energy, add it to the Data object as data.binding_energy
            if hasattr(data, 'binding_energy'):
                all_true_energies.extend(data.binding_energy.cpu().numpy())
                all_pred_energies.extend(output.cpu().numpy())
            else:
                # fallback: use y and output as proxy
                all_true_energies.extend(y.cpu().numpy())
                all_pred_energies.extend(output.cpu().numpy())
    
    # Now we can call the plotting function with both arguments
    print("Label distribution before conversion:", np.unique(all_labels, return_counts=True))
    all_labels = np.array(all_labels)
    if not set(np.unique(all_labels)).issubset({0, 1}):
        all_labels = (all_labels > 0.5).astype(int)

    plot_metrics_and_curves(
        train_losses, val_losses, 
        train_accs, val_accs,  # train_acc, val_acc
        all_preds, all_labels
    )

    # Plot predicted binding energy vs actual binding energy
    plt.figure(figsize=(6,6))
    plt.scatter(all_true_energies, all_pred_energies, alpha=0.6)
    plt.xlabel('Actual Binding Energy')
    plt.ylabel('Predicted Binding Energy')
    plt.title('Predicted vs Actual Binding Energy')
    plt.plot([min(all_true_energies), max(all_true_energies)], [min(all_true_energies), max(all_true_energies)], 'r--')
    plt.tight_layout()
    plt.savefig('pred_vs_actual_binding_energy.png')
    plt.show()
    