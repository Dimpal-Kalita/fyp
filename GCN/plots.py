import matplotlib.pyplot as plt
import torch
import numpy as np
from sklearn.metrics import roc_curve, auc

def plot_metrics_and_curves(train_losses, val_losses, train_acc, val_acc, all_preds, all_labels, train_rmsd=None, val_rmsd=None):
    """
    Plots training/validation loss, accuracy, and ROC curve for binary classification.
    """
    # Convert RMSD tensors to floats for plotting
    if train_rmsd is not None:
        train_rmsd = [t.detach().cpu().item() if hasattr(t, 'detach') else float(t) for t in train_rmsd]
    if val_rmsd is not None:
        val_rmsd = [t.detach().cpu().item() if hasattr(t, 'detach') else float(t) for t in val_rmsd]

    # Convert to numpy arrays
    if isinstance(all_labels, list):
        all_labels = np.array(all_labels)
    if isinstance(all_preds, list):
        all_preds = np.array(all_preds)

    plt.figure(figsize=(18, 5))

    # Subplot 1: Training and Validation Loss
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    # Optionally, plot RMSE curves if provided
    if train_rmsd is not None and val_rmsd is not None:
        plt.plot(train_rmsd, label='Train RMSE', linestyle='--')
        plt.plot(val_rmsd, label='Validation RMSE', linestyle='--')
        plt.legend()

    # Subplot 2: Training and Validation Accuracy
    plt.subplot(1, 3, 2)
    plt.plot(train_acc, label='Train Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    # Subplot 3: ROC Curve
    plt.subplot(1, 3, 3)
    fpr, tpr, _ = roc_curve(all_labels, all_preds)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig("plot.png")
    plt.show()


def plot_rmsd(rmsd_values):
    methods = ['AutoDock', 'Prediction Model', 'Vina']
    
    # Create the bar plot
    fig, ax = plt.subplots(figsize=(5, 3))
    bar_width = 0.1
    
    bars = ax.bar(methods, rmsd_values, bar_width, capsize=1, color='lightgreen', label='RMSD')
    
    # Add labels and title
    ax.set_ylabel('RMSD')
    ax.set_ylim(0, 6)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.legend()
    plt.tight_layout()
    plt.savefig("plot 2.png")
    plt.show()


