from Bio import PDB
import math

# Simple default parameters: you’ll want real values per bond type
DEFAULT_R0 = 1.5       # ideal bond length in Å
DEFAULT_K  = 300.0     # force constant in kJ/(mol·Å²)
BOND_CUTOFF = 2.0      # maximum distance to consider a bond (Å)

def calc_distance(a, b):
    """Euclidean distance between two Bio.PDB Atom objects."""
    coord_a = a.get_coord()
    coord_b = b.get_coord()
    return math.sqrt(((coord_a - coord_b)**2).sum())

def estimate_harmonic_energy(r, r0=DEFAULT_R0, k=DEFAULT_K):
    """Harmonic bond energy."""
    return 0.5 * k * (r - r0)**2

def parse_structure(file_path):
    parser    = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('protein', file_path)
    
    atoms = []
    # Collect all atoms in a flat list
    for model in structure:
        for chain in model:
            for res in chain:
                for atom in res:
                    atoms.append(atom)
    
    # Print out atom details
    print(f"Parsed {len(atoms)} atoms:\n")
    for atom in atoms:
        name     = atom.get_name()
        element  = atom.element
        coord    = atom.get_coord()
        occupancy= atom.get_occupancy()
        bfactor  = atom.get_bfactor()
        print(f"Atom {name:4s} ({element:2s}): "
              f"coord={coord.round(3).tolist()}, "
              f"occ={occupancy:.2f}, B={bfactor:.2f}")
    
    # Find bonds and compute energies
    print("\nDetected bonds (cutoff = "
          f"{BOND_CUTOFF} Å) and estimated energies:\n")
    n = len(atoms)
    for i in range(n):
        for j in range(i+1, n):
            a1 = atoms[i]
            a2 = atoms[j]
            dist = calc_distance(a1, a2)
            if dist <= BOND_CUTOFF:
                energy = estimate_harmonic_energy(dist)
                print(f"{a1.get_full_id()} -- {a2.get_full_id()} | "
                      f"r = {dist:.2f} Å, "
                      f"E_harm = {energy:.2f} kJ/mol")

if __name__ == "__main__":
    file_path = "GCN/data/1a1e/1a1e_complex_vina.pdb"
    parse_structure(file_path)
