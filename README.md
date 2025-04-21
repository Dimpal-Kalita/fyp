# Protein Structure Analysis with Graph Convolutional Networks

This project implements a memory-efficient Graph Convolutional Network (GCN) for analyzing protein structures. The model processes protein data represented as graphs and predicts binding properties.

## Project Overview

This Graph Convolutional Network (GCN) implementation is designed to analyze protein structures from PDB files. The model:

- Processes protein structures as graphs where atoms are nodes and bonds are edges
- Uses memory-efficient techniques to handle large protein structures
- Predicts binding properties with high accuracy
- Visualizes prediction results through various plots

## Requirements

The project requires Python 3.9+ and the following dependencies:

```
torch==2.6.0
torch-geometric==2.6.1
biopython==1.85
matplotlib==3.10.1
numpy==2.2.4
scikit-learn==1.6.1
networkx==3.4.2
tqdm==4.67.1
```

A complete list of dependencies is available in `requirements.txt`.

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd fyp
   ```

2. Install required packages:
   ```
   pip install -r requirements.txt
   ```

## Project Structure

```
/fyp
├── GCN/
│   ├── data/             # Directory containing protein structure data
│   ├── data_processing.py  # Data loading and preprocessing
│   ├── gnn_model.py        # Original GNN model implementation
│   ├── memory_efficient_gnn.py  # Memory-efficient GNN implementation
│   ├── GNN.py              # Main model training and evaluation module
│   ├── plots.py            # Plotting utilities
│   ├── prediction.py       # Model prediction functionality
│   ├── train.sh            # Training script
│   └── prediction.sh       # Prediction script
├── plot.png              # Example output plot
└── requirements.txt      # Project dependencies
```

## Usage

### Training the Model

To train the model, run:

```
cd fyp
sh GCN/train.sh
```

This will:
1. Load and preprocess protein data from the `GCN/data` directory
2. Split the data into training, validation, and test sets
3. Train the memory-efficient GNN model
4. Display training and validation metrics

### Making Predictions

To run predictions on protein structures:

```
cd fyp
sh GCN/prediction.sh
```

This will load a trained model and produce predictions for protein structures in the test set.

## Model Architecture

The implemented GNN model includes:

- Multiple GCN convolutional layers
- Memory optimization techniques:
  - Parameter sharing between layers
  - Progressive layer execution
  - Residual connections
  - Gradient checkpointing during training
- Global mean pooling for graph-level representations
- Fully connected output layers with sigmoid activation

## Results

The model performance metrics (accuracy, RMSD, ROC curve, etc.) are displayed during training and evaluation. Final plots are saved in the project root directory.

