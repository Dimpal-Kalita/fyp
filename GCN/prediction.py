# -*- coding: utf-8 -*-
"""Prediction Module"""

import torch
from torch_geometric.loader import DataLoader
from gnn_model import GNNModel
from memory_efficient_gnn import MemoryEfficientGNN
from data_processing import MolecularFeatureExtractor, ProteinLigandDataset, load_data_splits, load_data_split

def predict(model_path, test_loader, model_type="GCN", device=None):
    """
    Loads a trained model and makes predictions on the test dataset.
    
    Args:
        model_path: Path to the saved model
        test_loader: DataLoader containing test data
        model_type: Type of model to use ("GCN" or "MemoryEfficientGNN")
        device: Device to run the model on (CPU or GPU)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    # Load the model
    if model_type == "GCN":
        model = GNNModel().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
    elif model_type == "MemoryEfficientGNN":
        model = MemoryEfficientGNN().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
    else:
        raise NotImplementedError(f"Model type '{model_type}' is not implemented.")

    all_preds, all_labels = [], []

    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            output = model(data).squeeze()
            all_preds.extend(output.cpu().numpy())
            all_labels.extend(data.y.cpu().numpy())

    return all_preds, all_labels

if __name__ == "__main__":
    model_type = "GCN"  # Change to "MemoryEfficientGNN" to use the memory-efficient version
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if model_type in ["GCN", "MemoryEfficientGNN"]:
        data_dir = "./data"  # Path to PDB files
        _, _, test_data = load_data_split(data_dir)

        test_loader = DataLoader(test_data, batch_size=32)

        model_path = "./output/best_model.pt"  # Path to the saved model
        preds, labels = predict(model_path, test_loader, model_type, device)

        # Print results
        print("Predictions:", preds)
        print("Labels:", labels)
    else:
        # This part can be implemented if needed
        PROJECT_PATH = "Medusa_Graph/data/"
        MODEL_PATH = f"{PROJECT_PATH}/model_outputs/best_model.pt"

        splits = load_data_splits(PROJECT_PATH)
        feature_extractor = MolecularFeatureExtractor()

        test_loader = DataLoader(
            ProteinLigandDataset(splits, feature_extractor, split_type='test', data_fraction=0.5),
            batch_size=8, num_workers=0
        )

        preds, targets = predict(MODEL_PATH, test_loader, model_type, device)

        print("Predictions:", preds)
        print("Targets:", targets)

