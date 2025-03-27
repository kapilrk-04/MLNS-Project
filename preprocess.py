import os
import argparse
import pandas as pd
from tqdm import tqdm
import numpy as np
from scipy.spatial.distance import cdist

import torch
from torch_geometric.data import Data

from rdkit import Chem
from rdkit.Chem.rdmolops import GetAdjacencyMatrix

# print("Script started")

# --- Configuration ---
POSSIBLE_ATOM_TYPES = ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'I', 'B', 'Si', 'Se', 'H', 'Cd', 'Hg', 'Zn', 'Mn', 'Ca', 'Mg', 'Co', 'UNKNOWN']
POSSIBLE_HYBRIDIZATION = [Chem.rdchem.HybridizationType.SP,
                          Chem.rdchem.HybridizationType.SP2,
                          Chem.rdchem.HybridizationType.SP3,
                          Chem.rdchem.HybridizationType.SP3D,
                          Chem.rdchem.HybridizationType.SP3D2,
                          Chem.rdchem.HybridizationType.UNSPECIFIED,
                          'Unknown']
EDGE_FEATURE_DIM = 3
INTERACTION_CUTOFF = 4.5
# print("Configuration variables loaded")

# --- Helper Functions ---
def one_hot_encoding(value, possible_values):
    try:
        # print(f"OHE called for value: {value}")
        encoding = [0] * len(possible_values)
        try:
            # Only call strip if value is a string
            if isinstance(value, str):
                value_to_check = value.strip()
            else:
                value_to_check = value
                
            index = possible_values.index(value_to_check)
            encoding[index] = 1
        except ValueError:
            try:
                unknown_index = possible_values.index('Unknown')
                encoding[unknown_index] = 1
            except ValueError:
                encoding[-1] = 1
        return encoding
    except Exception as e:
        print(f"ERROR in one_hot_encoding: {e}")
        raise

def get_edge_features(edge_type):
    """
    Create edge features based on the type of connection.
    
    Parameters:
    -----------
    edge_type : str
        Type of edge: 'protein_bond', 'ligand_bond', or 'proximity'
        
    Returns:
    --------
    list
        Edge feature vector of dimension EDGE_FEATURE_DIM (3)
    """
    try:
        # print(f"Creating edge features for: {edge_type}")
        if edge_type == 'protein_bond':
            return [1.0, 0.0, 0.0]  # One-hot encoding for protein covalent bonds
        elif edge_type == 'ligand_bond':
            return [0.0, 1.0, 0.0]  # One-hot encoding for ligand covalent bonds
        elif edge_type == 'proximity':
            return [0.0, 0.0, 1.0]  # One-hot encoding for proximity-based interactions
        else:
            # Default fallback for unknown edge types
            return [0.0, 0.0, 0.0]
    except Exception as e:
        print(f"ERROR in get_edge_features: {e}")
        raise

def get_atom_features(atom, is_protein=False):
    try:
        # print(f"Getting atom features - is_protein: {is_protein}")
        atom_symbol = atom.GetSymbol() if atom else 'Unknown'
        # print(f"  Atom symbol: {atom_symbol}")
        atom_type = one_hot_encoding(atom_symbol, POSSIBLE_ATOM_TYPES)
        hybridization = one_hot_encoding(atom.GetHybridization(), POSSIBLE_HYBRIDIZATION) if atom else one_hot_encoding('Unknown', POSSIBLE_HYBRIDIZATION)
        num_h = min(atom.GetTotalNumHs(includeNeighbors=True), 10) if atom else 0
        formal_charge = max(-5, min(5, atom.GetFormalCharge())) if atom else 0
        is_aromatic = bool(atom.GetIsAromatic()) if atom else False
        is_in_ring = bool(atom.IsInRing()) if atom else False
        features = atom_type + hybridization + [num_h, formal_charge, int(is_aromatic), int(is_in_ring), float(is_protein)]
        # print(f"  Atom features generated successfully")
        return features
    except Exception as e:
        print(f"ERROR in get_atom_features: {e}")
        default_features = [0] * (len(POSSIBLE_ATOM_TYPES) + len(POSSIBLE_HYBRIDIZATION)) + [0, 0, 0, 0, float(is_protein)]
        try:
            default_features[POSSIBLE_ATOM_TYPES.index('Unknown')] = 1
        except ValueError:
            default_features[-1] = 1
        try:
            default_features[len(POSSIBLE_ATOM_TYPES) + POSSIBLE_HYBRIDIZATION.index('Unknown')] = 1
        except ValueError:
            default_features[len(POSSIBLE_ATOM_TYPES) + len(POSSIBLE_HYBRIDIZATION) - 1] = 1
        # print(f"  Using default features due to error")
        return default_features

def process_pdbbind_entry(pdb_id, protein_path, ligand_path, affinity_value, use_edge_attr=True):
    """Processes a single protein-ligand complex."""
    try:
        # print(f"\n=== Processing entry {pdb_id} ===")
        # print(f"Loading protein: {protein_path}")
        # print(f"Loading ligand: {ligand_path}")
        
        # 1. Load molecules
        protein_mol = Chem.MolFromPDBFile(protein_path, removeHs=False, sanitize=False) # Keep hydrogens if they exist
        # print(f"Protein loaded: {'Success' if protein_mol else 'Failed'}")
        
        # Try multiple formats for ligand loading with fallbacks
        ligand_mol = None
        if ligand_path.endswith('.mol2'):
            # print("Trying mol2 format")
            ligand_mol = Chem.MolFromMol2File(ligand_path, removeHs=False, sanitize=False)
        elif ligand_path.endswith('.sdf'):
            # print("Trying sdf format")
            # Try to read as an SD file (multiple molecules possible)
            sdf_supplier = Chem.SDMolSupplier(ligand_path, removeHs=False, sanitize=False)
            if sdf_supplier is not None and len(sdf_supplier) > 0:
                ligand_mol = sdf_supplier[0]  # Take the first molecule
        
        # Additional fallback for ligand: try as PDB if all else fails
        if ligand_mol is None and os.path.exists(ligand_path.replace('.mol2', '.pdb').replace('.sdf', '.pdb')):
            alt_path = ligand_path.replace('.mol2', '.pdb').replace('.sdf', '.pdb')
            # print(f"Trying alternative ligand format for {pdb_id}: {alt_path}")
            ligand_mol = Chem.MolFromPDBFile(alt_path, removeHs=False, sanitize=False)
        
        # print(f"Ligand loaded: {'Success' if ligand_mol else 'Failed'}")

        if protein_mol is None:
            print(f"Warning: Could not load protein {pdb_id} from {protein_path}")
            return None
        if ligand_mol is None:
            print(f"Warning: Could not load ligand {pdb_id} from {ligand_path}")
            return None
        
        # Try sanitizing - but continue if it fails
        # print("Attempting to sanitize molecules")
        try:
            Chem.SanitizeMol(protein_mol)
            # print("Protein sanitized successfully")
        except Exception as e:
            print(f"Warning: Could not sanitize protein {pdb_id}: {e}")
        
        try:
            Chem.SanitizeMol(ligand_mol)
            # print("Ligand sanitized successfully")
        except Exception as e:
            print(f"Warning: Could not sanitize ligand {pdb_id}: {e}")

        # 2. Extract Node Features and Coordinates
        # print("Extracting atoms and features")
        protein_atoms = protein_mol.GetAtoms()
        ligand_atoms = ligand_mol.GetAtoms()
        # print(f"Found {len(protein_atoms)} protein atoms and {len(ligand_atoms)} ligand atoms")

        node_features = []
        node_coords = []

        # Protein nodes
        # print("Processing protein atoms...")
        for atom_idx, atom in enumerate(protein_atoms):
            # if atom_idx % 500 == 0:
            #     # print(f"  Processing protein atom {atom_idx}/{len(protein_atoms)}")
            node_features.append(get_atom_features(atom, is_protein=True))
            try:
                pos = protein_mol.GetConformer().GetAtomPosition(atom.GetIdx())
                node_coords.append([pos.x, pos.y, pos.z])
            except:
                # If coordinates can't be found, use origin as fallback
                node_coords.append([0.0, 0.0, 0.0])
                # print(f"Warning: Missing coordinates for protein atom {atom.GetIdx()} in {pdb_id}")

        n_protein_atoms = len(protein_atoms)
        # print(f"Processed {n_protein_atoms} protein atoms")

        # Ligand nodes
        # print("Processing ligand atoms...")
        for atom_idx, atom in enumerate(ligand_atoms):
            # if atom_idx % 100 == 0:
            #     print(f"  Processing ligand atom {atom_idx}/{len(ligand_atoms)}")
            node_features.append(get_atom_features(atom, is_protein=False))
            try:
                pos = ligand_mol.GetConformer().GetAtomPosition(atom.GetIdx())
                node_coords.append([pos.x, pos.y, pos.z])
            except:
                # If coordinates can't be found, use origin as fallback
                node_coords.append([0.0, 0.0, 0.0])
                # print(f"Warning: Missing coordinates for ligand atom {atom.GetIdx()} in {pdb_id}")

        n_ligand_atoms = len(ligand_atoms)
        n_total_atoms = n_protein_atoms + n_ligand_atoms
        # print(f"Processed {n_ligand_atoms} ligand atoms")

        # Handle empty molecules
        if n_protein_atoms == 0 or n_ligand_atoms == 0:
            # print(f"Warning: Empty molecule found for {pdb_id}. Protein: {n_protein_atoms} atoms, Ligand: {n_ligand_atoms} atoms")
            return None

        # print("Creating node feature tensor")
        x = torch.tensor(node_features, dtype=torch.float)
        # print("Creating position tensor")
        pos = torch.tensor(node_coords, dtype=torch.float)
        # print(f"Node feature tensor shape: {x.shape}, Position tensor shape: {pos.shape}")

        # 3. Define Edges
        # print("Defining edges")
        edge_index_list = []
        edge_attr_list = [] # Store features for corresponding edges

        # Modified approach: Directly iterate through bonds instead of creating adjacency matrix
        # print("Creating protein bonds")
        try:
            # Intra-protein edges (Covalent bonds)
            bond_count = 0
            for bond in protein_mol.GetBonds():
                begin_idx = bond.GetBeginAtomIdx()
                end_idx = bond.GetEndAtomIdx()
                if begin_idx < end_idx:  # Avoid duplicates and self-loops
                    edge_index_list.append([begin_idx, end_idx])
                    if use_edge_attr: edge_attr_list.append(get_edge_features('protein_bond'))
                    bond_count += 1
            # print(f"Created {bond_count} protein bonds")
        except Exception as e:
            # print(f"Warning: Could not process protein bonds for {pdb_id}: {e}")
            # Fallback: create edges between consecutive atoms
            # print("Using fallback for protein bonds")
            for i in range(n_protein_atoms - 1):
                edge_index_list.append([i, i+1])
                if use_edge_attr: edge_attr_list.append(get_edge_features('protein_bond'))
            # print(f"Created {n_protein_atoms - 1} fallback protein bonds")

        # print("Creating ligand bonds")
        try:
            # Intra-ligand edges (Covalent bonds)
            bond_count = 0
            for bond in ligand_mol.GetBonds():
                begin_idx = bond.GetBeginAtomIdx() + n_protein_atoms  # Offset ligand indices
                end_idx = bond.GetEndAtomIdx() + n_protein_atoms      # Offset ligand indices
                if begin_idx < end_idx:  # Avoid duplicates
                    edge_index_list.append([begin_idx, end_idx])
                    if use_edge_attr: edge_attr_list.append(get_edge_features('ligand_bond'))
                    bond_count += 1
            # print(f"Created {bond_count} ligand bonds")
        except Exception as e:
            # print(f"Warning: Could not process ligand bonds for {pdb_id}: {e}")
            # Fallback: create edges between consecutive atoms
            # print("Using fallback for ligand bonds")
            for i in range(n_protein_atoms, n_total_atoms - 1):
                edge_index_list.append([i, i+1])
                if use_edge_attr: edge_attr_list.append(get_edge_features('ligand_bond'))
            # print(f"Created {n_ligand_atoms - 1} fallback ligand bonds")

        # Inter-molecular edges (Proximity-based)
        # print("Creating proximity-based interactions")
        try:
            if n_protein_atoms > 0 and n_ligand_atoms > 0:
                # print("Computing pairwise distances")
                protein_coords = pos[:n_protein_atoms].numpy()
                ligand_coords = pos[n_protein_atoms:].numpy()

                # Efficiently compute pairwise distances
                # print(f"Protein coords shape: {protein_coords.shape}, Ligand coords shape: {ligand_coords.shape}")
                dist_matrix = cdist(protein_coords, ligand_coords)
                # print(f"Distance matrix shape: {dist_matrix.shape}")

                # Find pairs within cutoff
                # print(f"Finding interactions within {INTERACTION_CUTOFF} Å")
                interaction_indices = np.where(dist_matrix <= INTERACTION_CUTOFF)
                protein_interact_idx = interaction_indices[0]
                ligand_interact_idx = interaction_indices[1] + n_protein_atoms # Offset ligand indices
                # print(f"Found {len(protein_interact_idx)} proximity interactions")

                for idx, (prot_idx, lig_idx) in enumerate(zip(protein_interact_idx, ligand_interact_idx)):
                    # if idx % 1000 == 0:
                    #     # print(f"  Processing proximity interaction {idx}/{len(protein_interact_idx)}")
                    edge_index_list.append([prot_idx, lig_idx])
                    if use_edge_attr: edge_attr_list.append(get_edge_features('proximity'))
        except Exception as e:
            # print(f"Warning: Could not compute proximity edges for {pdb_id}: {e}")
            # Fallback: add a small number of connections between protein and ligand
            # print("Using fallback for proximity interactions")
            edge_index_list.append([0, n_protein_atoms])  # Connect first atom of each molecule
            if use_edge_attr: edge_attr_list.append(get_edge_features('proximity'))

        if not edge_index_list: # Handle cases with no edges found
             # print(f"Warning: No edges found for {pdb_id}. Creating minimal connectivity model.")
             # Create at least one edge to ensure graph connectivity
             if n_protein_atoms > 0 and n_ligand_atoms > 0:
                 edge_index_list.append([0, n_protein_atoms])  # Connect first atom of protein to first atom of ligand
                 if use_edge_attr: edge_attr_list.append(get_edge_features('proximity'))

        # print(f"Creating edge tensors from {len(edge_index_list)} edges")
        # Combine and make undirected
        edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
        # Add reverse edges
        edge_index_undirected = torch.cat([edge_index, edge_index.flip(dims=[0])], dim=1)
        # print(f"Edge index tensor shape: {edge_index_undirected.shape}")

        edge_attr = None
        if use_edge_attr and edge_attr_list:
            # print("Creating edge attribute tensor")
            edge_attr_tensor = torch.tensor(edge_attr_list, dtype=torch.float)
            # Duplicate attributes for reverse edges
            edge_attr = torch.cat([edge_attr_tensor, edge_attr_tensor], dim=0)
            # print(f"Edge attr tensor shape: {edge_attr.shape}")
            
            # Make sure dimensions match
            if edge_index_undirected.shape[1] != edge_attr.shape[0]:
                # print(f"Warning: Edge index and attribute dimension mismatch in {pdb_id}. Adjusting...")
                # print(f"  Edge index shape: {edge_index_undirected.shape}, Edge attr shape: {edge_attr.shape}")
                # Create a correct size edge_attr tensor
                edge_attr = edge_attr.repeat(edge_index_undirected.shape[1] // edge_attr.shape[0] + 1, 1)[:edge_index_undirected.shape[1]]
                # print(f"  New edge attr shape: {edge_attr.shape}")

        # 4. Target Value
        # print(f"Setting target value: {affinity_value}")
        y = torch.tensor([affinity_value], dtype=torch.float)

        # 5. Create Data object
        # print("Creating PyG Data object")
        data = Data(x=x, edge_index=edge_index_undirected, edge_attr=edge_attr, y=y, pos=pos, pdb_id=pdb_id) # Include pos and id
        # print(f"Data object created successfully for {pdb_id}")

        return data

    except Exception as e:
        # print(f"Error processing {pdb_id}: {e}")
        import traceback
        traceback.print_exc() # Print detailed traceback
        return None

# --- Main Script ---
if __name__ == "__main__":
    # print("\n=== SCRIPT EXECUTION STARTED ===\n")
    parser = argparse.ArgumentParser(description="Preprocess PDBbind dataset for Graph Neural Networks.")
    parser.add_argument("pdbbind_dir", type=str, help="Root directory of the PDBbind dataset (containing folder and index files).")
    parser.add_argument("output_dir", type=str, help="Directory to save the processed PyG Data objects (.pt files).")
    parser.add_argument("--index_file", type=str, default="", help="Name of the index file to use (e.g., INDEX_general_PL_data.2016 or INDEX_refined_data.2016).")
    parser.add_argument("--data_subdir", type=str, default="", help="Subdirectory within pdbbind_dir containing the PDB/MOL2 files (e.g., 'v2016' or 'refined-set').")
    parser.add_argument("--no_edge_attr", action="store_true", help="Do not compute or store edge attributes.")
    parser.add_argument("--sample_size", type=int, default=0, help="Process only a sample of entries (0 for all entries).")

    print("Parsing arguments")
    args = parser.parse_args()

    pdbbind_root = args.pdbbind_dir
    output_root = args.output_dir
    index_filename = args.index_file
    data_folder = args.data_subdir
    use_edge_attributes = not args.no_edge_attr
    sample_size = args.sample_size

    # print(f"Arguments parsed successfully:")
    # print(f"  PDBbind root: {pdbbind_root}")
    # print(f"  Output directory: {output_root}")
    # print(f"  Index file: {index_filename}")
    # print(f"  Data subfolder: {data_folder}")
    # print(f"  Use edge attributes: {use_edge_attributes}")
    # print(f"  Sample size: {sample_size}")

    # Create output directory if it doesn't exist
    print(f"Creating output directory: {output_root}")
    os.makedirs(output_root, exist_ok=True)

    # --- Load Index File ---
    index_path = os.path.join(pdbbind_root, index_filename)
    # print(f"Loading index file from: {index_path}")
    
    try:
        # First try to read the entire file as text to handle any irregular lines
        # print("Reading index file")
        with open(index_path, 'r') as f:
            lines = f.readlines()
        
        print(f"Read {len(lines)} lines from index file")
        # Filter out comment lines and empty lines
        data_lines = [line.strip() for line in lines if line.strip() and not line.strip().startswith('#')]
        print(f"Found {len(data_lines)} data lines after filtering")
        
        # Parse each line manually to handle inconsistencies
        data = []
        # print("Parsing data lines")
        for i, line in enumerate(data_lines):
            # if i % 1000 == 0:
            #     # print(f"  Parsing line {i}/{len(data_lines)}")
            # Split by whitespace for the fixed-width columns
            parts = line.split()
            
            # We know the first 4 columns are fixed format
            if len(parts) >= 4:
                pdb_id = parts[0]
                resolution = parts[1]
                release_year = parts[2]
                neg_log_affinity = parts[3]
                
                # Combine the rest as it may contain spaces within fields
                remaining = ' '.join(parts[4:])
                
                # Try to extract the affinity value units and reference
                try:
                    affinity = float(neg_log_affinity)
                    row = {
                        'pdb_id': pdb_id,
                        'resolution': resolution, 
                        'release_year': release_year,
                        'neg_log_affinity': neg_log_affinity,
                        'remaining': remaining,
                        'affinity': affinity
                    }
                    data.append(row)
                except ValueError:
                    print(f"Warning: Could not parse affinity value '{neg_log_affinity}' for {pdb_id}")
            else:
                print(f"Warning: Line doesn't have enough columns: {line}")
        
        # Create DataFrame from the parsed data
        # print(f"Creating DataFrame from {len(data)} parsed entries")
        index_df = pd.DataFrame(data)
        
        # Drop entries where affinity couldn't be parsed
        # print("Dropping entries with missing affinity values")
        index_df.dropna(subset=['affinity'], inplace=True)
        
    except Exception as e:
        print(f"Error reading index file {index_path}: {e}")
        exit(1)

    # print(f"Found {len(index_df)} entries with valid affinity values in index file.")
    
    # Sample a subset if requested
    if sample_size > 0 and sample_size < len(index_df):
        # print(f"Sampling {sample_size} entries")
        index_df = index_df.sample(sample_size, random_state=42)
        # print(f"Sampled {sample_size} entries for processing")

    # --- Process Entries ---
    processed_count = 0
    skipped_count = 0
    error_count = 0
    # print(f"Starting processing. Outputting to: {output_root}")

    for idx, (_, row) in enumerate(tqdm(index_df.iterrows(), total=len(index_df), desc="Processing PDBbind entries")):
        try:
            # print(f"\n--- Processing entry {idx+1}/{len(index_df)} ---")
            pdb_id = row['pdb_id']
            affinity = row['affinity']
            # print(f"PDB ID: {pdb_id}, Affinity: {affinity}")

            protein_file = os.path.join(pdbbind_root, data_folder, pdb_id, f"{pdb_id}_protein.pdb")
            ligand_file = os.path.join(pdbbind_root, data_folder, pdb_id, f"{pdb_id}_ligand.mol2") # Common format

            if not os.path.exists(protein_file):
                # print(f"Warning: Protein file not found for {pdb_id} at {protein_file}. Skipping.")
                skipped_count += 1
                continue
                
            if not os.path.exists(ligand_file):
                # Try SDF as fallback if MOL2 fails
                ligand_file_sdf = os.path.join(pdbbind_root, data_folder, pdb_id, f"{pdb_id}_ligand.sdf")
                if os.path.exists(ligand_file_sdf):
                    ligand_file = ligand_file_sdf
                    # print(f"Using SDF format for ligand: {ligand_file_sdf}")
                else:
                    # print(f"Warning: Ligand file not found for {pdb_id} at {ligand_file} (or .sdf). Skipping.")
                    skipped_count += 1
                    continue

            # Process the entry
            # print(f"Calling process_pdbbind_entry for {pdb_id}")
            data_object = process_pdbbind_entry(pdb_id, protein_file, ligand_file, affinity, use_edge_attr=use_edge_attributes)

            # Save the processed data
            if data_object is not None:
                output_path = os.path.join(output_root, f"{pdb_id}.pt")
                # print(f"Saving data object to {output_path}")
                torch.save(data_object, output_path)
                processed_count += 1
                # print(f"Successfully saved {pdb_id}.pt")
            else:
                # print(f"Skipping {pdb_id} due to processing failure")
                skipped_count += 1
        except Exception as e:
            print(f"Critical error processing entry {row['pdb_id'] if 'pdb_id' in row else 'unknown'}: {e}")
            import traceback
            traceback.print_exc()
            error_count += 1
            # Continue instead of letting the exception propagate
            continue

    print("\n--- Preprocessing Complete ---")
    print(f"Successfully processed: {processed_count} entries.")
    print(f"Skipped/Failed: {skipped_count} entries.")
    print(f"Errors: {error_count} entries.")
    print(f"Processed data saved in: {output_root}")
    print("\n=== SCRIPT EXECUTION COMPLETED ===")