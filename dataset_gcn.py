# --- START OF FILE dataset_gcn.py ---
import os
import pandas as pd
import numpy as np
import pickle
from scipy.spatial import distance_matrix
import multiprocessing
from itertools import repeat
import networkx as nx
import torch
from torch.utils.data import Dataset
from rdkit import Chem
from rdkit import RDLogger
from torch_geometric.data import Batch, Data
import torch_geometric.utils as pyg_utils
import warnings

RDLogger.DisableLog('rdApp.*')
np.set_printoptions(threshold=np.inf)
warnings.filterwarnings('ignore')

# --- Helper functions (one_of_k_encoding, one_of_k_encoding_unk, atom_features from dataset_GIGN_modified.py) ---
def one_of_k_encoding(k, possible_values):
    if k not in possible_values:
        raise ValueError(f"{k} is not a valid value in {possible_values}")
    return [k == e for e in possible_values]

def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def atom_features(mol, graph, atom_symbols=['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'I', 'H', 'B', 'Si', 'Se', 'Te'], explicit_H=False): # Expanded symbols
    """
    Generates node features for atoms in the molecule and adds them to the nx graph.
    Note: explicit_H=False matches RDKit's default behavior after removeHs=True.
    If removeHs=False in preprocessing, then explicit_H should be True here.
    """
    for atom in mol.GetAtoms():
        results = one_of_k_encoding_unk(atom.GetSymbol(), atom_symbols + ['Unknown']) + \
                one_of_k_encoding_unk(atom.GetDegree(),[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) + \
                one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6]) + \
                one_of_k_encoding_unk(atom.GetHybridization(), [
                    Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                    Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
                    Chem.rdchem.HybridizationType.SP3D2, Chem.rdchem.HybridizationType.UNSPECIFIED,
                    Chem.rdchem.HybridizationType.S, Chem.rdchem.HybridizationType.OTHER # Added S and OTHER
                    ]) + [atom.GetIsAromatic()] + \
                one_of_k_encoding_unk(atom.GetFormalCharge(), [-3, -2, -1, 0, 1, 2, 3]) # Expanded formal charge

        if explicit_H: # Only add if explicit_H is True (meaning Hs were NOT removed by RDKit)
            results = results + one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])
        else: # If Hs were removed, TotalNumHs might be misleading. Add placeholder or based on implicit.
             # For removeHs=True, GetTotalNumHs typically counts implicit Hs if not 0.
             # It might be better to rely on implicit valence or just not add this feature if Hs are removed.
             # For simplicity, let's keep it if explicit_H is False, assuming RDKit handles it.
            results = results + one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])


        atom_feats = np.array(results).astype(np.float32)
        graph.add_node(atom.GetIdx(), feats=torch.from_numpy(atom_feats))

def get_intra_edge_index(mol, graph):
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        graph.add_edge(i, j)

def mol_to_nx_graph(mol):
    graph = nx.Graph()
    atom_features(mol, graph, explicit_H=False) # Assuming removeHs=True in preprocessing
    get_intra_edge_index(mol, graph)
    return graph

def get_inter_edge_index(ligand, pocket, dis_threshold = 5.):
    atom_num_l = ligand.GetNumAtoms()
    pos_l = ligand.GetConformers()[0].GetPositions()
    pos_p = pocket.GetConformers()[0].GetPositions()
    dis_matrix = distance_matrix(pos_l, pos_p)
    node_idx = np.where(dis_matrix < dis_threshold)
    inter_edges = []
    for i, j in zip(node_idx[0], node_idx[1]):
        inter_edges.append((i, atom_num_l + j))
        inter_edges.append((atom_num_l + j, i))
    if not inter_edges:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.LongTensor(inter_edges).t().contiguous()

def complex_to_pyg_graph_gcn(complex_path, label, save_path, dis_threshold=5.):
    try:
        with open(complex_path, 'rb') as f:
            ligand, pocket = pickle.load(f)
    except Exception as e:
        print(f"Error loading complex {complex_path}: {e}")
        return

    if ligand is None or pocket is None or ligand.GetNumAtoms() == 0 or pocket.GetNumAtoms() == 0:
        print(f"Skipping complex {complex_path} due to missing or empty ligand/pocket.")
        return

    atom_num_l = ligand.GetNumAtoms()
    atom_num_p = pocket.GetNumAtoms()

    graph_l = mol_to_nx_graph(ligand)
    graph_p = mol_to_nx_graph(pocket)

    x_l = torch.stack([feat['feats'] for _, feat in graph_l.nodes(data=True)]) if graph_l.number_of_nodes() > 0 else torch.empty((0,0)) # Handle empty graph
    x_p = torch.stack([feat['feats'] for _, feat in graph_p.nodes(data=True)]) if graph_p.number_of_nodes() > 0 else torch.empty((0,0))
    
    if x_l.nelement() == 0 and x_p.nelement() == 0 :
        print(f"Skipping {complex_path} due to no features in ligand and pocket")
        return
    elif x_l.nelement() == 0:
        x = x_p
    elif x_p.nelement() == 0:
        x = x_l
    else:
        # Ensure feature dimensions match before concat if one is empty and other is not
        if x_l.shape[1] != x_p.shape[1] and x_l.nelement() > 0 and x_p.nelement() > 0:
             print(f"Feature dimension mismatch in {complex_path}. Ligand: {x_l.shape}, Pocket: {x_p.shape}. Skipping.")
             return
        x = torch.cat([x_l, x_p], dim=0)


    edge_index_l_intra = pyg_utils.from_networkx(graph_l).edge_index if graph_l.number_of_nodes() > 0 else torch.empty((2,0), dtype=torch.long)
    edge_index_p_intra = (pyg_utils.from_networkx(graph_p).edge_index + atom_num_l) if graph_p.number_of_nodes() > 0 else torch.empty((2,0), dtype=torch.long)
    edge_index_intra = torch.cat([edge_index_l_intra, edge_index_p_intra], dim=-1)
    edge_index_inter = get_inter_edge_index(ligand, pocket, dis_threshold=dis_threshold)

    combined_edge_index = torch.cat([edge_index_intra, edge_index_inter], dim=1)
    num_intra = edge_index_intra.size(1)
    num_inter = edge_index_inter.size(1)
    attr_intra = torch.tensor([[1, 0]] * num_intra, dtype=torch.float)
    attr_inter = torch.tensor([[0, 1]] * num_inter, dtype=torch.float)
    combined_edge_attr = torch.cat([attr_intra, attr_inter], dim=0)

    if combined_edge_index.size(1) != combined_edge_attr.size(0) and combined_edge_index.size(1) > 0:
       print(f"Warning: Mismatch in edge index ({combined_edge_index.size(1)}) and attr ({combined_edge_attr.size(0)}) for {complex_path}.")
       return

    y_val = float(label)
    y = torch.FloatTensor([y_val])

    data = Data(x=x, edge_index=combined_edge_index, edge_attr=combined_edge_attr, y=y)
    torch.save(data, save_path)


class GCNGraphDataset(Dataset):
    def __init__(self, data_dir, data_df, dis_threshold=5, graph_type='Graph_GCN',
                 num_process=8, create=False, complex_id_col='pdbid', target_col='-logkd/ki'): # Note target_col
        self.data_dir = data_dir
        self.data_df = data_df
        self.dis_threshold = dis_threshold
        self.graph_type = graph_type
        self.complex_id_col = complex_id_col
        self.target_col = target_col
        self.graph_paths = []
        self.complex_ids = []
        self.n_features = None
        self.num_edge_features = None

        print("Preparing list of graph file paths for GCN...")
        valid_indices = []
        complex_rdkit_path_list = []
        target_list = []
        graph_pyg_path_list = [] # For storing paths during creation

        for i, row in data_df.iterrows():
            cid = str(row[self.complex_id_col]) # Ensure cid is string
            try:
                target_val = float(row[self.target_col])
            except (ValueError, TypeError):
                print(f"Warning: Invalid target value for {cid} ('{row[self.target_col]}'). Skipping.")
                continue

            complex_dir = os.path.join(data_dir, cid)
            complex_rdkit_path = os.path.join(complex_dir, f"{cid}_{self.dis_threshold}A.rdkit")
            graph_pyg_path = os.path.join(complex_dir, f"{graph_type}-{cid}_{self.dis_threshold}A.pyg")

            if create:
                if os.path.exists(complex_rdkit_path):
                    complex_rdkit_path_list.append(complex_rdkit_path)
                    target_list.append(target_val)
                    graph_pyg_path_list.append(graph_pyg_path) # Store path for saving
                    # Add to self.graph_paths for loading after creation
                    self.graph_paths.append(graph_pyg_path)
                    self.complex_ids.append(cid)
                    valid_indices.append(i)
                # else:
                    # print(f"Warning: Source .rdkit file {complex_rdkit_path} not found for creation. Skipping {cid}.")
            else:
                if os.path.exists(graph_pyg_path):
                    self.graph_paths.append(graph_pyg_path)
                    self.complex_ids.append(cid)
                    valid_indices.append(i)

        self.data_df = self.data_df.iloc[valid_indices].reset_index(drop=True)
        print(f"Found {len(self.graph_paths)} potential graph files for GCN.")

        if create and complex_rdkit_path_list:
            print(f'Starting GCN graph generation for {len(complex_rdkit_path_list)} complexes using {num_process} processes...')
            pool = multiprocessing.Pool(num_process)
            try:
                pool.starmap(complex_to_pyg_graph_gcn,
                             zip(complex_rdkit_path_list, target_list, graph_pyg_path_list, repeat(self.dis_threshold, len(target_list))))
            except Exception as e:
                print(f"Error during multiprocessing GCN graph creation: {e}")
            finally:
                pool.close()
                pool.join()
            print("GCN graph generation complete.")
            # Update graph_paths to only those successfully created if some failed (more robust)
            self.graph_paths = [p for p in self.graph_paths if os.path.exists(p)]
            print(f"{len(self.graph_paths)} GCN graph files available after creation.")


        if self.graph_paths:
            try:
                sample_graph = torch.load(self.graph_paths[0])
                self.n_features = sample_graph.x.shape[-1]
                if hasattr(sample_graph, 'edge_attr') and sample_graph.edge_attr is not None:
                    self.num_edge_features = sample_graph.edge_attr.shape[-1]
                else:
                    self.num_edge_features = 0
                print(f"GCN Dataset: Input node features: {self.n_features}, Edge features: {self.num_edge_features}")
            except Exception as e:
                print(f"Error loading sample GCN graph {self.graph_paths[0]} to determine features: {e}")
                self.n_features = -1
                self.num_edge_features = -1
        else:
            print("No GCN graph paths found, cannot determine feature sizes.")


    def __len__(self):
        return len(self.graph_paths)

    def __getitem__(self, index):
        try:
            data = torch.load(self.graph_paths[index])
            # Ensure all necessary attributes are present for the GCN model
            if not hasattr(data, 'x') or not hasattr(data, 'edge_index') or not hasattr(data, 'y'):
                 print(f"Graph {self.graph_paths[index]} is missing essential attributes. Skipping.")
                 return None
            # edge_attr is optional for GCN but good for GINE
            if not hasattr(data, 'edge_attr'):
                 data.edge_attr = None # Or create a default if your GCN/GINE expects it
            return data
        except Exception as e:
             print(f"Error loading GCN graph at index {index}, path {self.graph_paths[index]}: {e}")
             return None

    @staticmethod
    def collate_fn(batch):
        batch = [data for data in batch if data is not None]
        if not batch:
            return None
        return Batch.from_data_list(batch)

if __name__ == '__main__':
    data_root_main = 'data' # Main data folder
    # Adjusted path to match the preprocessing script's output structure
    data_dir_main = os.path.join(data_root_main, 'PDBbind_v2020_other_PL/v2020-other-PL')
    csv_path_main = os.path.join(data_root_main, 'binding_data.csv')
    target_col_main = '-logkd/ki' # Match column name in preprocessing and CSV
    complex_id_col_main = 'pdbid'

    if not os.path.exists(csv_path_main):
         print(f"Error: CSV file not found at {csv_path_main}")
    elif not os.path.isdir(data_dir_main):
         print(f"Error: Data directory not found at {data_dir_main}")
    else:
        data_df_main = pd.read_csv(csv_path_main)
        # Ensure target column exists
        if target_col_main not in data_df_main.columns:
            print(f"Error: Target column '{target_col_main}' not found in {csv_path_main}.")
            print(f"Available columns: {data_df_main.columns.tolist()}")
        else:
            print("--- Creating GCN graphs (if they don't exist) ---")
            gcn_dataset_create = GCNGraphDataset(
                data_dir_main, data_df_main, graph_type='Graph_GCN',
                create=True, num_process=4, # Set create=True
                complex_id_col=complex_id_col_main, target_col=target_col_main
            )
            print(f"GCN Dataset size after creation attempt: {len(gcn_dataset_create)}")

            print("\n--- Loading GCN graphs for use ---")
            gcn_dataset_load = GCNGraphDataset(
                data_dir_main, data_df_main, graph_type='Graph_GCN',
                create=False, # Set create=False for subsequent runs
                complex_id_col=complex_id_col_main, target_col=target_col_main
            )
            print(f"GCN Dataset size for loading: {len(gcn_dataset_load)}")

            if len(gcn_dataset_load) > 0 and gcn_dataset_load.n_features > 0:
                 from torch_geometric.loader import DataLoader
                 train_loader = DataLoader(gcn_dataset_load, batch_size=4, shuffle=True,
                                           num_workers=0, collate_fn=GCNGraphDataset.collate_fn)
                 print(f"\nGCN DataLoader created with {len(train_loader)} batches.")
                 try:
                    first_batch = next(iter(train_loader))
                    if first_batch:
                        print("\nSample GCN batch loaded successfully:")
                        print(first_batch)
                        print("Batch x shape:", first_batch.x.shape)
                        print("Batch edge_index shape:", first_batch.edge_index.shape)
                        if first_batch.edge_attr is not None:
                             print("Batch edge_attr shape:", first_batch.edge_attr.shape)
                        else:
                             print("Batch edge_attr: None")
                        print("Batch y shape:", first_batch.y.shape)
                    else:
                        print("\nFailed to load the first GCN batch.")
                 except StopIteration:
                     print("\nGCN DataLoader is empty.")
                 except Exception as e:
                     print(f"\nError iterating GCN DataLoader: {e}")
            else:
                print("\nCannot create GCN DataLoader, dataset is empty or features not determined.")
# --- END OF FILE dataset_gcn.py ---