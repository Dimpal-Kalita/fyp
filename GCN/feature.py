import os
import io
import numpy as np
from Bio import PDB
from Bio.PDB import PDBIO
from rdkit import Chem
from rdkit.Chem import AllChem
import torch
from torch_geometric.data import Data
import re
from Bio.PDB.PDBIO import Select

# ----------------------------------------------
# configuration
# ----------------------------------------------
noncovalent_cutoff = 8.0  # Å
element_types = ['c', 'n', 'o', 's', 'p', 'cl', 'br', 'i', 'h']

DOCKING_SCORE_MIN = -12.0
DOCKING_SCORE_MAX = -9.0

# ----------------------------------------------
# utility functions
# ----------------------------------------------

def one_hot_encoding(x, allowed_values):
    return [1 if x == v else 0 for v in allowed_values]

def normalize_score(score, min_val=DOCKING_SCORE_MIN, max_val=DOCKING_SCORE_MAX):
    norm = (score - min_val) / (max_val - min_val)
    return min(max(norm, 0.0), 1.0)  # clip to [0, 1]

def parse_docking_score(pdb_path):
    score = None
    with open(pdb_path, 'r') as f:
        for line in f:
            if 'REMARK VINA RESULT' in line:
                parts = line.strip().split()
                for tok in parts:
                    if re.match(r'[-+]?\d*\.\d+|\d+', tok):
                        try:
                            score = float(tok)
                            return score
                        except ValueError:
                            continue
            if 'Estimated Free Energy of Binding' in line:
                m = re.search(r'([-+]?\d*\.\d+|\d+)', line)
                if m:
                    score = float(m.group(1))
                    return score
    if score is None:
        raise ValueError(f"No docking score found in {pdb_path}")
    return score

# ----------------------------------------------
# step 1: load pdb and separate atoms
# ----------------------------------------------

def load_structure(pdb_path):
    parser = PDB.PDBParser(QUIET=True)
    return parser.get_structure('complex', pdb_path)

def split_ligand_protein(structure):
    ligand_res = []
    protein_res = []
    for model in structure:
        for chain in model:
            for residue in chain:
                hetflag = residue.get_id()[0]
                resname = residue.get_resname()
                if resname.lower() == 'unk':
                    ligand_res.append(residue)
                elif resname.lower() != 'hoh':
                    protein_res.append(residue)
    ligand_atoms = [a for res in ligand_res for a in res]
    protein_atoms = [a for res in protein_res for a in res]
    return ligand_atoms, protein_atoms

# ----------------------------------------------
# step 2: create rdkit molecule from ligand
# ----------------------------------------------

class LigandSelect(Select):
    def __init__(self, ligand_residues):
        self.ligand_ids = {(res.get_parent().id, res.get_resname(), res.id) for res in ligand_residues}
    def accept_atom(self, atom):
        res = atom.get_parent()
        return (res.get_parent().id, res.get_resname(), res.id) in self.ligand_ids

def rdkit_from_ligand(structure, ligand_residues):
    pdb_io = PDBIO()
    pdb_io.set_structure(structure)
    buf = io.StringIO()
    pdb_io.save(buf, LigandSelect(ligand_residues))
    pdb_block = buf.getvalue()

    processed_lines = []
    for line in pdb_block.splitlines():
        if line.startswith("ATOM") or line.startswith("HETATM"):
            line = "HETATM" + line[6:]
            line = line.ljust(80)
            std_line = (
                line[0:6] + line[6:11] + line[12:16] + line[16:20] + line[20:22] +
                line[22:26] + line[30:38] + line[38:46] + line[46:54] + line[76:78] + "\n"
            )
            processed_lines.append(std_line)
        elif line.strip() == "TER":
            processed_lines.append("TER\n")
    processed_block = "".join(processed_lines)

    mol = Chem.MolFromPDBBlock(processed_block, removeHs=False)
    if mol is None:
        raise ValueError("rdkit failed to parse ligand pdb block.")
    return mol

# ----------------------------------------------
# step 3: atomic features
# ----------------------------------------------

def atom_features(atom, rdkit_atom=None):
    elem = atom.element if atom.element else atom.get_name()[0]
    one_hot = one_hot_encoding(elem, element_types)
    atomic_num = Chem.GetPeriodicTable().GetAtomicNumber(elem) if elem in element_types else 0
    coord = atom.get_coord().astype(np.float32)
    charge = 0.0
    heavy_neighbors = 0
    if rdkit_atom:
        try:
            charge = float(rdkit_atom.GetProp('_GasteigerCharge'))
        except Exception:
            charge = 0.0
        heavy_neighbors = len(rdkit_atom.GetNeighbors())
    feats = one_hot + [atomic_num, charge, heavy_neighbors]
    return coord, feats

# ----------------------------------------------
# step 4: build edges and global features
# ----------------------------------------------

def build_edges(coords):
    n = len(coords)
    edge_index = []
    edge_attr = []
    for i in range(n):
        for j in range(i+1, n):
            dist = np.linalg.norm(coords[i] - coords[j])
            if dist <= noncovalent_cutoff:
                edge_index += [[i, j], [j, i]]
                edge_attr += [[dist], [dist]]
    ei = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    ea = torch.tensor(edge_attr, dtype=torch.float)
    return ei, ea

def compute_global(lig_coords, prot_coords):
    com_l = np.mean(lig_coords, axis=0)
    com_p = np.mean(prot_coords, axis=0)
    com_dist = np.linalg.norm(com_l - com_p)
    contacts = sum(
        1 for lc in lig_coords for pc in prot_coords
        if np.linalg.norm(lc - pc) <= noncovalent_cutoff
    )
    return [com_dist, contacts]

# ----------------------------------------------
# main featurization pipeline
# ----------------------------------------------

def featurize_complex(pdb_path, docking_score=None):
    if docking_score is None:
        docking_score = parse_docking_score(pdb_path)
    struct = load_structure(pdb_path)
    ligand_atoms, protein_atoms = split_ligand_protein(struct)
    ligand_residues = list({a.get_parent() for a in ligand_atoms})
    lig_mol = rdkit_from_ligand(struct, ligand_residues)
    AllChem.ComputeGasteigerCharges(lig_mol)

    atoms = ligand_atoms + protein_atoms
    coords_list, feat_list = [], []

    lig_coords = [a.get_coord() for a in ligand_atoms]
    rdkit_map = {}
    for idx, atom in enumerate(ligand_atoms):
        for j in range(lig_mol.GetNumAtoms()):
            pos = lig_mol.GetConformer().GetAtomPosition(j)
            if np.allclose([pos.x, pos.y, pos.z], atom.get_coord(), atol=1e-3):
                rdkit_map[idx] = j
                break

    for i, atom in enumerate(atoms):
        rd_atom = None
        if i < len(ligand_atoms) and i in rdkit_map:
            rd_atom = lig_mol.GetAtomWithIdx(rdkit_map[i])
        coord, feats = atom_features(atom, rd_atom)
        coords_list.append(coord)
        feat_list.append(feats)

    coords_arr = np.stack(coords_list)
    feats_arr = np.array(feat_list, dtype=np.float32)

    edge_index, edge_attr = build_edges(coords_arr)
    global_feats = compute_global(
        coords_arr[:len(ligand_atoms)],
        coords_arr[len(ligand_atoms):]
    )

    normalized_score = 1 if normalize_score(docking_score) > 0.5 else 0
    data_obj = Data(
        x=torch.tensor(feats_arr),
        pos=torch.tensor(coords_arr),
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=torch.tensor([normalized_score], dtype=torch.float),
        u=torch.tensor(global_feats, dtype=torch.float)
    )
    return data_obj

# ----------------------------------------------
# example usage
# ----------------------------------------------

if __name__ == '__main__':
    for fname in ['1a1e_complex_vina.pdb', '1a1e_complex_ad4.pdb']:
        data_obj = featurize_complex(f'gcn/data/1a1e/{fname}')
        print(fname, data_obj.y.item())
