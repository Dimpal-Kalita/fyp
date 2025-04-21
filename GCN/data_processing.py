# -*- coding: utf-8 -*-
"""Data Processing Module"""

import os
import random
import pickle
import numpy as np
from tqdm import tqdm
from torch_geometric.data import Data, Dataset
import torch
from torch_geometric.data import Data
from Bio import PDB
from sklearn.model_selection import train_test_split

ELEMENTS = {
    # Define atomic numbers for elements
    "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9,
    "Ne": 10, "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15, "S": 16, "Cl": 17, "Ar": 18,
    "K": 19, "Ca": 20, "Sc": 21, "Ti": 22, "V": 23, "Cr": 24, "Mn": 25, "Fe": 26, 
    "Co": 27, "Ni": 28, "Cu": 29, "Zn": 30, "Ga": 31, "Ge": 32, "As": 33, "Se": 34,
    "Br": 35, "Kr": 36, "Rb": 37, "Sr": 38, "Y": 39, "Zr": 40, "Nb": 41, "Mo": 42,
    "Tc": 43, "Ru": 44, "Rh": 45, "Pd": 46, "Ag": 47, "Cd": 48, "In": 49, "Sn": 50,
    "Sb": 51, "Te": 52, "I": 53, "Xe": 54, "Cs": 55, "Ba": 56, "La": 57, "Ce": 58,
    "Pr": 59, "Nd": 60, "Pm": 61, "Sm": 62, "Eu": 63, "Gd": 64, "Tb": 65, "Dy": 66,
    "Ho": 67, "Er": 68, "Tm": 69, "Yb": 70, "Lu": 71, "Hf": 72, "Ta": 73, "W": 74,
    "Re": 75, "Os": 76, "Ir": 77, "Pt": 78, "Au": 79, "Hg": 80, "Tl": 81, "Pb": 82,
    "Bi": 83, "Po": 84, "At": 85, "Rn": 86, "Fr": 87, "Ra": 88, "Ac": 89, "Th": 90,
    "Pa": 91, "U": 92, "Np": 93, "Pu": 94, "Am": 95, "Cm": 96, "Bk": 97, "Cf": 98,
    "Es": 99, "Fm": 100, "Md": 101, "No": 102, "Lr": 103, "Rf": 104, "Db": 105,
    "Sg": 106, "Bh": 107, "Hs": 108, "Mt": 109, "Ds": 110, "Rg": 111, "Cn": 112,
    "Nh": 113, "Fl": 114, "Mc": 115, "Lv": 116, "Ts": 117, "Og": 118
}


# Set random seeds for reproducibility
def set_seeds(seed=42):
    """
    Set random seeds for reproducibility across numpy, random, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

class MolecularFeatureExtractor:
    """
    Extracts molecular features such as atom type encoding, mass, electronegativity,
    and spatial coordinates from molecular complexes.
    """
    def __init__(self):
        # Define atom type mappings
        self.atom_types = ['C', 'O', 'N', 'H', 'S', 'P', 'ZN', 'CA', 'MG', 'CL', 'F']
        self.atomic_masses = {
            'C': 12.01, 'O': 16.00, 'N': 14.01, 'H': 1.008, 'S': 32.07,
            'P': 30.97, 'ZN': 65.38, 'CA': 40.08, 'MG': 24.31, 'CL': 35.45, 'F': 19.00
        }
        self.electronegativities = {
            'C': 2.55, 'O': 3.44, 'N': 3.04, 'H': 2.20, 'S': 2.58,
            'P': 2.19, 'ZN': 1.65, 'CA': 1.00, 'MG': 1.31, 'CL': 3.16, 'F': 3.98
        }

    def get_atom_features(self, atom_type, is_ligand, coords):
        """
        Extract features for a single atom, including encoded type,
        normalized mass, electronegativity, and coordinates.
        """
        atom_type_enc = [1 if atom_type == t else 0 for t in self.atom_types]
        mass = self.atomic_masses.get(atom_type, 0.0)
        electroneg = self.electronegativities.get(atom_type, 0.0)
        mass_norm = np.log1p(mass) / np.log1p(max(self.atomic_masses.values()))
        electroneg_norm = electroneg / max(self.electronegativities.values())

        features = (
            atom_type_enc +
            [mass_norm, electroneg_norm, float(is_ligand),
             coords[0], coords[1], coords[2]]
        )
        return np.array(features, dtype=np.float32)

    def process_complex(self, complex_data):
        """
        Processes a molecular complex (PDB file content) to extract node and edge features.
        """
        atoms, is_ligand, coords = [], [], []
        for line in complex_data.split('\n'):
            if line.startswith(('ATOM', 'HETATM')):
                atom_type = line[76:78].strip()
                residue = line[17:20].strip()
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])

                atoms.append(atom_type)
                is_ligand.append(residue == 'UNK')
                coords.append([x, y, z])

        coords = np.array(coords)
        node_features = [
            self.get_atom_features(atom, lig, coord)
            for atom, lig, coord in zip(atoms, is_ligand, coords)
        ]

        edge_index, edge_features = [], []
        distance_cutoff = 4.5
        for i in range(len(coords)):
            for j in range(i + 1, len(coords)):
                distance = np.linalg.norm(coords[i] - coords[j])
                if distance < distance_cutoff:
                    edge_index.extend([[i, j], [j, i]])
                    edge_feat = np.array([
                        distance, 1.0 / distance, np.exp(-distance),
                        float(is_ligand[i] and is_ligand[j]),
                        float(not is_ligand[i] and not is_ligand[j]),
                        float(is_ligand[i] != is_ligand[j])
                    ])
                    edge_features.extend([edge_feat, edge_feat])

        return {
            'node_features': np.array(node_features),
            'edge_index': np.array(edge_index).T,
            'edge_features': np.array(edge_features),
            'coords': coords
        }

class ProteinLigandDataset(Dataset):
    """
    Custom dataset for protein-ligand complexes, enabling PyTorch Geometric compatibility.
    """
    def __init__(self, split_data, feature_extractor, split_type='train', transform=None, data_fraction=0.5):
        super().__init__(transform)
        all_metadata = split_data[split_type]['metadata']
        num_samples = int(len(all_metadata) * data_fraction)
        sampled_metadata = random.sample(all_metadata, num_samples)

        self.split_data = {
            'metadata': sampled_metadata,
            'ad4': split_data[split_type]['ad4'][:num_samples],
            'vina': split_data[split_type]['vina'][:num_samples]
        }
        self.feature_extractor = feature_extractor
        self.transform = transform
        self.processed_data = []

        print(f"Processing {split_type} dataset ({data_fraction*100}% of data)...")
        self._process_complexes()

    def _process_complexes(self):
        """
        Processes molecular complexes into graph data for GNNs.
        """
        for item in tqdm(self.split_data['metadata']):
            pdb_id = item['pdb_id']
            try:
                complex_path = os.path.join("Medusa_Graph/data", pdb_id, f"{pdb_id}_complex_ad4.pdb")
                with open(complex_path, 'r') as f:
                    complex_data = f.read()

                features = self.feature_extractor.process_complex(complex_data)
                data = Data(
                    x=torch.FloatTensor(features['node_features']),
                    edge_index=torch.LongTensor(features['edge_index']),
                    edge_attr=torch.FloatTensor(features['edge_features']),
                    y=torch.FloatTensor([item['vina_score']]),
                    pos=torch.FloatTensor(features['coords'])
                )

                if self.transform:
                    data = self.transform(data)

                self.processed_data.append(data)
            except Exception as e:
                print(f"Error processing {pdb_id}: {str(e)}")
                continue

    def len(self):
        return len(self.processed_data)

    def get(self, idx):
        return self.processed_data[idx]

def load_data_splits(project_path):
    """
    Load pre-split data or raise an error if data processing has not been performed.
    """
    splits_path = os.path.join(project_path, 'data_splits.pkl')
    if os.path.exists(splits_path):
        with open(splits_path, 'rb') as f:
            return pickle.load(f)
    else:
        raise FileNotFoundError("Data splits not found. Please run data processing first.")

def pdb_to_graph(file_path):
    """
    Converts a PDB file into a graph representation.
    Nodes represent atoms, and edges are based on distance thresholds.
    """
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('protein', file_path)
    atoms = list(structure.get_atoms())

    node_features = []
    edge_index = []

    # Node features: atomic number for each atom
    for atom in atoms:
        atomic_number = ELEMENTS.get(atom.element, 0)  # Use 0 if element is not found
        node_features.append([atomic_number])

    # Edges based on distance threshold (e.g., 3.5 Å)
    for i, atom_i in enumerate(atoms):
        for j, atom_j in enumerate(atoms):
            if i != j:
                distance = atom_i - atom_j
                if distance < 3.5:
                    edge_index.append([i, j])

    x = torch.tensor(node_features, dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    # Example target (binary classification label)
    y = torch.tensor([random.randint(0, 1)], dtype=torch.float)  # Replace with actual label if available

    return Data(x=x, edge_index=edge_index, y=y)

def load_data_split(data_dir, train_ratio=0.7, val_ratio=1.5, test_ratio=1.5):
    """
    Loads and splits PDB files into train, validation, and test datasets.
    """
    all_files = [os.path.join(root, file) for root, _, files in os.walk(data_dir) for file in files if file.endswith(".pdb")]

    if not all_files:
        raise ValueError("No PDB files found in the specified directory.")
    
    train_files, temp_files = train_test_split(all_files, train_size=train_ratio)
    val_files, test_files = train_test_split(temp_files, test_size=test_ratio / (test_ratio + val_ratio))
    
    train_data = [pdb_to_graph(f) for f in train_files]
    val_data = [pdb_to_graph(f) for f in val_files]
    test_data = [pdb_to_graph(f) for f in test_files]
    
    return train_data, val_data, test_data