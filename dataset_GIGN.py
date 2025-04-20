# %%
import os
import pandas as pd
import numpy as np
import pickle
from scipy.spatial import distance_matrix
import multiprocessing
from itertools import repeat
import networkx as nx
import torch 
from torch.utils.data import Dataset # Standard Dataset
# from torch_geometric.data import DataLoader # Using PyG's DataLoader alias
from torch_geometric.loader import DataLoader as PyGDataLoader # Explicit PyG DataLoader
from rdkit import Chem
from rdkit import RDLogger
from rdkit import Chem
from torch_geometric.data import Batch, Data
import warnings
from tqdm import tqdm

RDLogger.DisableLog('rdApp.*')
np.set_printoptions(threshold=np.inf)
warnings.filterwarnings('ignore')

# %%
def one_of_k_encoding(k, possible_values):
    if k not in possible_values:
        raise ValueError(f"{k} is not a valid value in {possible_values}")
    return [k == e for e in possible_values]


def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

# Atom symbols used for one-hot encoding
ATOM_SYMBOLS = ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'I', 'Unknown']
ATOM_DEGREES = [0, 1, 2, 3, 4, 5, 6]
ATOM_IMPLICIT_VALENCES = [0, 1, 2, 3, 4, 5, 6]
ATOM_HYBRIDIZATIONS = [
    Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2, Chem.rdchem.HybridizationType.UNSPECIFIED # Added for robustness
]
ATOM_TOTAL_HS = [0, 1, 2, 3, 4]


def atom_features(mol, graph, explicit_H=True):
    for atom in mol.GetAtoms():
        results = one_of_k_encoding_unk(atom.GetSymbol(), ATOM_SYMBOLS[:-1]) + \
                one_of_k_encoding_unk(atom.GetDegree(), ATOM_DEGREES) + \
                one_of_k_encoding_unk(atom.GetImplicitValence(), ATOM_IMPLICIT_VALENCES) + \
                one_of_k_encoding_unk(atom.GetHybridization(), ATOM_HYBRIDIZATIONS[:-1]) + \
                [atom.GetIsAromatic()]
        if explicit_H:
            results = results + one_of_k_encoding_unk(atom.GetTotalNumHs(), ATOM_TOTAL_HS)

        atom_feats = np.array(results).astype(np.float32)
        graph.add_node(atom.GetIdx(), feats=torch.from_numpy(atom_feats))

def get_edge_index_and_attr(mol, graph): # Modified to add edge_attr (dummy for now)
    # For this example, edge_attr will be a placeholder if use_edge_attr is True in model
    # A real implementation would extract bond features here.
    # Placeholder: edge_attr = [bond_type_is_single, bond_type_is_double, ...]
    # Let's use a fixed dimension for dummy edge_attr, e.g., 3
    dummy_edge_feature_dim = 3 
    
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        graph.add_edge(i, j)
        # Add dummy edge attributes for the example
        # In a real scenario, extract actual bond features here
        # graph.edges[i,j]['attr'] = torch.randn(dummy_edge_feature_dim) # For one direction
        # graph.edges[j,i]['attr'] = torch.randn(dummy_edge_feature_dim) # For other direction if graph becomes directed

def mol2graph(mol, add_dummy_edge_attr=False):
    graph = nx.Graph()
    atom_features(mol, graph)
    get_edge_index_and_attr(mol, graph) # Modified call

    x = torch.stack([feats['feats'] for n, feats in graph.nodes(data=True)])
    
    # Convert to directed to get both (i,j) and (j,i) if GNNs expect that
    # Or keep as undirected and let PyG Data handle to_undirected / is_undirected later.
    # For GIGN, edge_index_intra and inter are typically directed (both ways) from its processing.
    # Let's make them explicitly directed here.
    directed_graph = nx.DiGraph(graph) # Create a directed version with edges in both directions
    
    if directed_graph.edges():
        edge_index_list = []
        edge_attr_list = []
        for u, v, data in directed_graph.edges(data=True):
            edge_index_list.append(torch.LongTensor([u,v]))
            if add_dummy_edge_attr:
                 # Example: create dummy edge features of dim 3
                # bond = mol.GetBondBetweenAtoms(u, v) # RDKit Mol, not nx graph
                # if bond:
                #     bt = bond.GetBondTypeAsDouble()
                #     dummy_attr = torch.tensor([bt == 1.0, bt == 2.0, bt == 0.0], dtype=torch.float) # single, double, other
                # else: # Should not happen if edge exists
                dummy_attr = torch.zeros(3, dtype=torch.float) # Fallback
                edge_attr_list.append(dummy_attr)

        edge_index = torch.stack(edge_index_list).T
        edge_attr = torch.stack(edge_attr_list) if add_dummy_edge_attr and edge_attr_list else None
    else:
        edge_index = torch.empty((2,0), dtype=torch.long)
        edge_attr = None if not add_dummy_edge_attr else torch.empty((0,3), dtype=torch.float)
        
    return x, edge_index, edge_attr


def inter_graph(ligand, pocket, dis_threshold = 5., add_dummy_edge_attr=False):
    atom_num_l = ligand.GetNumAtoms()
    
    graph_inter = nx.Graph() # Start as undirected
    pos_l = ligand.GetConformers()[0].GetPositions()
    pos_p = pocket.GetConformers()[0].GetPositions()

    if pos_l.shape[0] == 0 or pos_p.shape[0] == 0: # No atoms in ligand or pocket
        return torch.empty((2,0), dtype=torch.long), (torch.empty((0,3), dtype=torch.float) if add_dummy_edge_attr else None)

    dis_matrix = distance_matrix(pos_l, pos_p)
    node_idx = np.where(dis_matrix < dis_threshold)
    for i, j in zip(node_idx[0], node_idx[1]):
        graph_inter.add_edge(i, j + atom_num_l) # i is ligand index, j is pocket index (shifted)
        # Add dummy edge attributes if needed
        # if add_dummy_edge_attr:
        #     graph_inter.edges[i, j + atom_num_l]['attr'] = torch.randn(3) # Placeholder

    directed_graph_inter = nx.DiGraph(graph_inter) # Make directed (edges both ways)

    if directed_graph_inter.edges():
        edge_index_list = []
        edge_attr_list = []
        for u,v,data in directed_graph_inter.edges(data=True):
            edge_index_list.append(torch.LongTensor([u,v]))
            if add_dummy_edge_attr:
                # Dummy edge features for inter-graph edges (e.g., distance based, or just type)
                dummy_attr = torch.ones(3, dtype=torch.float) # Different dummy for inter
                edge_attr_list.append(dummy_attr)

        edge_index_inter = torch.stack(edge_index_list).T
        edge_attr_inter = torch.stack(edge_attr_list) if add_dummy_edge_attr and edge_attr_list else None
    else:
        edge_index_inter = torch.empty((2,0), dtype=torch.long)
        edge_attr_inter = None if not add_dummy_edge_attr else torch.empty((0,3), dtype=torch.float)
        
    return edge_index_inter, edge_attr_inter

# %%
def mols2graphs(complex_path, label, save_path, dis_threshold=5., add_dummy_edge_attr=False):
    try:
        if not os.path.isfile(complex_path):
            return

        with open(complex_path, 'rb') as f:
            ligand, pocket = pickle.load(f)

        if ligand is None or pocket is None or ligand.GetNumConformers() == 0 or pocket.GetNumConformers() == 0 \
           or ligand.GetNumAtoms() == 0 : # Pocket can be 0 atoms if far
            # print(f"[SKIPPED] Invalid ligand/pocket or no conformers/atoms in: {complex_path}")
            return

        atom_num_l = ligand.GetNumAtoms()
        atom_num_p = pocket.GetNumAtoms()

        pos_l = torch.FloatTensor(ligand.GetConformers()[0].GetPositions())
        x_l, edge_index_l, edge_attr_l = mol2graph(ligand, add_dummy_edge_attr)
        
        if atom_num_p > 0:
            pos_p = torch.FloatTensor(pocket.GetConformers()[0].GetPositions())
            x_p, edge_index_p, edge_attr_p = mol2graph(pocket, add_dummy_edge_attr)
            x = torch.cat([x_l, x_p], dim=0)
            pos = torch.cat([pos_l, pos_p], dim=0)
            split = torch.cat([torch.zeros(atom_num_l, dtype=torch.long), torch.ones(atom_num_p, dtype=torch.long)], dim=0)
            edge_index_p_shifted = edge_index_p + atom_num_l if edge_index_p.numel() > 0 else torch.empty((2,0), dtype=torch.long)
            edge_index_intra = torch.cat([edge_index_l, edge_index_p_shifted], dim=-1)
            
            edge_attr_list_intra = []
            if add_dummy_edge_attr:
                if edge_attr_l is not None: edge_attr_list_intra.append(edge_attr_l)
                if edge_attr_p is not None: edge_attr_list_intra.append(edge_attr_p)
            edge_attr_intra = torch.cat(edge_attr_list_intra, dim=0) if add_dummy_edge_attr and edge_attr_list_intra else None

        else: # No pocket atoms
            x = x_l
            pos = pos_l
            split = torch.zeros(atom_num_l, dtype=torch.long)
            edge_index_intra = edge_index_l
            edge_attr_intra = edge_attr_l if add_dummy_edge_attr else None
            
        edge_index_inter, edge_attr_inter = inter_graph(ligand, pocket, dis_threshold=dis_threshold, add_dummy_edge_attr=add_dummy_edge_attr)
        
        # Combined edge_index and edge_attr for general GNN layers
        list_to_cat_idx = []
        list_to_cat_attr = []

        if edge_index_intra.numel() > 0: list_to_cat_idx.append(edge_index_intra)
        if edge_index_inter.numel() > 0: list_to_cat_idx.append(edge_index_inter)
        
        if add_dummy_edge_attr:
            if edge_attr_intra is not None: list_to_cat_attr.append(edge_attr_intra)
            if edge_attr_inter is not None: list_to_cat_attr.append(edge_attr_inter)

        edge_index = torch.cat(list_to_cat_idx, dim=-1) if list_to_cat_idx else torch.empty((2,0), dtype=torch.long)
        edge_attr = torch.cat(list_to_cat_attr, dim=0) if add_dummy_edge_attr and list_to_cat_attr else None


        y = torch.FloatTensor([label])
        
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                    edge_index_intra=edge_index_intra, 
                    edge_index_inter=edge_index_inter, 
                    y=y, pos=pos, split=split)
        
        # Validate data object (example checks)
        if data.x is None or data.x.shape[0] == 0: return
        if data.edge_index.numel() > 0 and (data.edge_index.max() >= data.x.shape[0]): return # Invalid edge index


        torch.save(data, save_path)
    except Exception as e:
        print(f"[ERROR] Failed to process {complex_path}: {e}")


class PLIDataLoader(PyGDataLoader): # Use PyG's DataLoader
    def __init__(self, data, **kwargs):
        # Custom collate_fn is passed to PyGDataLoader if needed, or use its default.
        # The dataset's collate_fn is usually for torch.utils.data.DataLoader.
        # PyG's Batch.from_data_list is the default collate_fn for PyGDataLoader.
        # Our dataset's __getitem__ already returns Data objects.
        super().__init__(data, **kwargs) # Removed custom collate_fn, PyG handles it


class GraphDataset(Dataset):
    def __init__(self, data_dir, data_df, dis_threshold=5, graph_type='Graph_GIGN', 
                 num_process=8, create=False, add_dummy_edge_attr=False): # Added add_dummy_edge_attr
        self.data_dir = data_dir
        self.data_df = data_df
        self.dis_threshold = dis_threshold
        self.graph_type = graph_type
        self.create = create
        self.add_dummy_edge_attr = add_dummy_edge_attr # Store this
        self.graph_paths = []
        self.complex_ids = [] 
        self.num_process = num_process
        self._pre_process()

    def _pre_process(self):
        complex_paths_to_process = []
        pKa_values = []
        graph_save_paths = []
        dis_thresholds_list = []
        add_dummy_edge_attr_list = [] # For starmap

        print("Preparing file list for processing...")
        for i, row in tqdm(self.data_df.iterrows(), total=len(self.data_df), desc="Scanning CSV"):
            cid = str(row['pdbid'])
            if cid == 'index' or cid == 'readme': continue
            try:
                pKa = float(row['-logkd/ki'])
                complex_specific_dir = os.path.join(self.data_dir, cid)
                rdkit_complex_path = os.path.join(complex_specific_dir, f"{cid}_{self.dis_threshold}A.rdkit")
                graph_pyg_path = os.path.join(complex_specific_dir, f"{self.graph_type}-{cid}_{self.dis_threshold}A.pyg")

                if self.create:
                    if os.path.exists(rdkit_complex_path):
                        # If .pyg exists, skip generation unless forced (not implemented here)
                        # if not os.path.exists(graph_pyg_path): 
                        complex_paths_to_process.append(rdkit_complex_path)
                        pKa_values.append(pKa)
                        graph_save_paths.append(graph_pyg_path)
                        dis_thresholds_list.append(self.dis_threshold)
                        add_dummy_edge_attr_list.append(self.add_dummy_edge_attr)
            except Exception as e:
                print(f"Skipping {cid} due to error during scan: {e}")
        
        if self.create and complex_paths_to_process:
            print(f"Generating {len(complex_paths_to_process)} complex graphs with {self.num_process} processes...")
            with multiprocessing.Pool(self.num_process) as pool:
                pool.starmap(mols2graphs,
                             zip(complex_paths_to_process, pKa_values, graph_save_paths, 
                                 dis_thresholds_list, add_dummy_edge_attr_list)) # Added add_dummy_edge_attr_list
            print("Graph generation complete.")

        print("Verifying existing graph files...")
        final_graph_paths = []
        final_complex_ids = []
        for i, row in tqdm(self.data_df.iterrows(), total=len(self.data_df), desc="Finalizing graph list"):
            cid = str(row['pdbid'])
            if cid == 'index' or cid == 'readme': continue
            complex_specific_dir = os.path.join(self.data_dir, cid)
            graph_pyg_path = os.path.join(complex_specific_dir, f"{self.graph_type}-{cid}_{self.dis_threshold}A.pyg")
            if os.path.exists(graph_pyg_path):
                try:
                    # Minimal load check
                    # test_data = torch.load(graph_pyg_path)
                    # if not all(hasattr(test_data, attr) for attr in ['x', 'y', 'pos']): # Basic check
                    #    raise ValueError("Missing essential attributes")
                    final_graph_paths.append(graph_pyg_path)
                    final_complex_ids.append(cid)
                except Exception as load_e:
                    print(f"Could not verify/load .pyg file {graph_pyg_path}, skipping: {load_e}")
        
        self.graph_paths = final_graph_paths
        self.complex_ids = final_complex_ids
        print(f"Dataset initialized with {len(self.graph_paths)} graphs.")
        if not self.graph_paths:
            print("Warning: No graphs found. Check paths, .rdkit files, and 'create' flag.")


    def __len__(self):
        return len(self.graph_paths)

    def __getitem__(self, idx):
        try:
            data = torch.load(self.graph_paths[idx])
            # Add PDB ID if needed by downstream tasks
            data.pdb_id = self.complex_ids[idx]
            
            # --- CRITICAL VALIDATION ---
            if data.x is None or data.x.shape[0] == 0:
                # print(f"Warning: Graph {self.complex_ids[idx]} has no node features. Skipping.")
                return None
            if data.edge_index.numel() > 0 and data.edge_index.max() >= data.x.shape[0]:
                # print(f"Warning: Graph {self.complex_ids[idx]} has invalid edge indices. Skipping.")
                return None
            if not hasattr(data, 'pos') or data.pos is None or data.pos.shape[0] != data.x.shape[0]:
                # print(f"Warning: Graph {self.complex_ids[idx]} has missing or mismatched 'pos'. Skipping.")
                return None
            if not hasattr(data, 'y') or data.y is None:
                 # print(f"Warning: Graph {self.complex_ids[idx]} has no target 'y'. Skipping.")
                return None
            if not hasattr(data, 'edge_index_intra') or not hasattr(data, 'edge_index_inter'):
                # print(f"Warning: Graph {self.complex_ids[idx]} missing GIGN edge indices. Skipping.")
                return None

            # Ensure edge_attr exists if add_dummy_edge_attr was true, else ensure it's None
            if self.add_dummy_edge_attr:
                if not hasattr(data, 'edge_attr') or data.edge_attr is None:
                    # Create dummy if missing but expected
                    data.edge_attr = torch.zeros((data.edge_index.shape[1], 3), dtype=torch.float) 
            else: # Not adding dummy, so ensure it's not there or set to None
                if hasattr(data, 'edge_attr') and data.edge_attr is not None:
                    # This case implies add_dummy_edge_attr was False during creation, but files have it.
                    # For consistency with model expectation, could set to None if model configured for no edge_attr.
                    # Or, it implies a mismatch in how dataset was created vs. loaded.
                    # For now, let's assume if add_dummy_edge_attr is False, model doesn't expect edge_attr.
                    pass # Let it pass through, model config will determine usage.
            return data
        except Exception as e:
            # print(f"Error loading graph {self.complex_ids[idx]} (path: {self.graph_paths[idx]}): {e}. Skipping.")
            return None

    # PyGDataLoader uses Batch.from_data_list by default, so a custom collate_fn here
    # is not strictly necessary if __getitem__ returns Data objects and handles None.
    # However, if we want to filter Nones from a batch before PyG's collate, it's useful.
    @staticmethod
    def collate_fn(batch):
        batch = [item for item in batch if item is not None]
        if not batch:
            return None 
        return Batch.from_data_list(batch)


if __name__ == '__main__':
    data_root_example = './data' 
    complex_dir_example = os.path.join(data_root_example, 'PDBbind_v2020_other_PL/v2020-other-PL')
    csv_path_example = os.path.join(data_root_example, 'binding_data.csv')

    if not (os.path.exists(complex_dir_example) and os.path.exists(csv_path_example)):
        print("Please ensure ./data/PDBbind_v2020_other_PL/v2020-other-PL and ./data/binding_data.csv exist.")
        print("And that preprocess.py has been run to generate .rdkit files in complex directories.")
    else:
        example_df = pd.read_csv(csv_path_example)
        example_df_subset = example_df.head(10) # Small subset for testing

        print(f"Test with {len(example_df_subset)} complexes.")
        # Set create=True to attempt generation of .pyg files from .rdkit files
        # Set add_dummy_edge_attr=True if your model will use edge_attr
        example_dataset = GraphDataset(data_dir=complex_dir_example, 
                                       data_df=example_df_subset, 
                                       graph_type='Graph_GIGN_Example', 
                                       create=True, 
                                       num_process=1, # Use 1 for easier debugging initially
                                       add_dummy_edge_attr=True) # Test with dummy edge_attr

        if len(example_dataset) > 0:
            example_loader = PLIDataLoader(example_dataset, batch_size=2, shuffle=False, collate_fn=GraphDataset.collate_fn)
            for i, batch_data_example in enumerate(example_loader):
                if batch_data_example is None:
                    print(f"Batch {i} is None.")
                    continue
                print(f"Batch {i}: Num graphs: {batch_data_example.num_graphs}, x: {batch_data_example.x.shape}, edge_attr: {batch_data_example.edge_attr.shape if batch_data_example.edge_attr is not None else None}")
                break 
        else:
            print("Example dataset is empty. Check .rdkit files and paths.")