import numpy as np
import torch
from torch_geometric.data import Data
from Bio import PDB
import re

ELEMENTS = {
    "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9,
    "Ne": 10, "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15, "S": 16, "Cl": 17, "Ar": 18,
    # ... (add more as needed)
}

def extract_vina_score(pdb_content):
    """
    Extracts the Vina score from a PDB file content if present (e.g., in REMARK lines).
    Handles various formats, e.g.:
    - REMARK VINA RESULT: -7.5 0.0 0.0
    - REMARK   VINA RESULT    -8.2
    - REMARK VINA RESULT:   -6.1
    - REMARK VINA RESULT -5.7
    """
    for line in pdb_content.split('\n'):
        if 'VINA RESULT' in line:
            # Try to find a floating point number after 'VINA RESULT' (with or without colon, with any whitespace)
            match = re.search(r'VINA RESULT[:\s]*([-+]?[0-9]*\.?[0-9]+)', line)
            if match:
                try:
                    return float(match.group(1))
                except Exception:
                    continue
    return None

def pdb_to_graph_features(file_path):
    """
    Parses a PDB file and extracts x, edge_index, edge_attr, pos, and y.
    - x: node features (17 features: 11 one-hot atom types, normalized mass, electronegativity, is_ligand, x, y, z)
    - edge_index: edges based on distance threshold
    - edge_attr: edge features (distance)
    - pos: coordinates
    - y: label (binary: 1 if Vina score < -7.0, else 0)
    """
    atom_types = ['C', 'O', 'N', 'H', 'S', 'P', 'ZN', 'CA', 'MG', 'CL', 'F']
    atomic_masses = {
        'C': 12.01, 'O': 16.00, 'N': 14.01, 'H': 1.008, 'S': 32.07,
        'P': 30.97, 'ZN': 65.38, 'CA': 40.08, 'MG': 24.31, 'CL': 35.45, 'F': 19.00
    }
    electronegativities = {
        'C': 2.55, 'O': 3.44, 'N': 3.04, 'H': 2.20, 'S': 2.58,
        'P': 2.19, 'ZN': 1.65, 'CA': 1.00, 'MG': 1.31, 'CL': 3.16, 'F': 3.98
    }
    max_mass = np.log1p(max(atomic_masses.values()))
    max_electroneg = max(electronegativities.values())

    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('protein', file_path)
    atoms = list(structure.get_atoms())
    
    node_features = []
    coords = []
    for atom in atoms:
        atom_type = atom.element.upper()
        coord = atom.get_coord()
        # One-hot encoding for atom type
        atom_type_enc = [1 if atom_type == t else 0 for t in atom_types]
        # Normalized mass and electronegativity
        mass = atomic_masses.get(atom_type, 0.0)
        electroneg = electronegativities.get(atom_type, 0.0)
        mass_norm = np.log1p(mass) / max_mass if max_mass > 0 else 0.0
        electroneg_norm = electroneg / max_electroneg if max_electroneg > 0 else 0.0
        # is_ligand: set to 0 (unknown)
        is_ligand = 0.0
        features = atom_type_enc + [mass_norm, electroneg_norm, is_ligand, coord[0], coord[1], coord[2]]
        node_features.append(features)
        coords.append(coord)
    coords = np.array(coords)

    edge_index = []
    edge_attr = []
    distance_cutoff = 4.5
    for i, coord_i in enumerate(coords):
        for j, coord_j in enumerate(coords):
            if i != j:
                distance = np.linalg.norm(coord_i - coord_j)
                if distance < distance_cutoff:
                    edge_index.append([i, j])
                    edge_attr.append([distance])

    x = torch.tensor(node_features, dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    pos = torch.tensor(coords, dtype=torch.float)

    # Extract y (Vina score) if available
    with open(file_path, 'r') as f:
        pdb_content = f.read()
    vina_score = extract_vina_score(pdb_content)
    # Binarize: 1 if vina_score < -8.5, else 0
    if vina_score is not None:
        y = torch.tensor([1.0 if vina_score < -8.5 else 0.0], dtype=torch.float)
    else:
        y = torch.tensor([float('nan')], dtype=torch.float)
    print(file_path, vina_score)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, pos=pos, y=y) 