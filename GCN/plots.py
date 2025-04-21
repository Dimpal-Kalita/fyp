import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve

def plot_metrics_and_curves(train_losses, val_losses, train_acc, val_acc, all_preds, all_labels):
    """
    Plots training/validation loss and accuracy, ROC curve, precision-recall curve, 
    and predicted vs actual labels.
    """
    plt.figure(figsize=(15, 10))

    # Subplot 1: Training and Validation Loss and Accuracy
    plt.subplot(2, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.plot(train_acc, label='Train Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Loss / Accuracy')
    plt.title('Training and Validation Metrics')
    plt.legend()

    # Subplot 2: ROC Curve
    plt.subplot(2, 2, 2)
    fpr, tpr, _ = roc_curve(all_labels, all_preds)
    plt.plot(fpr, tpr, color='b', label='ROC curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc='lower right')

    # Subplot 3: Precision-Recall Curve
    plt.subplot(2, 2, 3)
    precision, recall, _ = precision_recall_curve(all_labels, all_preds)
    plt.plot(recall, precision, color='b', label='Precision-Recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='lower left')

    # Subplot 4: Predicted vs Actual Binding Affinity
    plt.subplot(2, 2, 4)
    plt.scatter(all_labels, all_preds)
    plt.xlabel("Actual Binding Affinity")
    plt.ylabel("Predicted Binding Affinity")
    plt.title("Predicted vs Actual Binding Affinity")

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


