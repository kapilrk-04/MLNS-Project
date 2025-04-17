# # %%
# import os
# import pandas as pd
# import numpy as np
# import pickle
# from scipy.spatial import distance_matrix
# import multiprocessing
# from itertools import repeat
# import networkx as nx
# import torch 
# from torch.utils.data import Dataset, DataLoader
# from rdkit import Chem
# from rdkit import RDLogger
# from rdkit import Chem
# from torch_geometric.data import Batch, Data
# import warnings
# from tqdm import tqdm
# RDLogger.DisableLog('rdApp.*')
# np.set_printoptions(threshold=np.inf)
# warnings.filterwarnings('ignore')

# # %%
# def one_of_k_encoding(k, possible_values):
#     if k not in possible_values:
#         raise ValueError(f"{k} is not a valid value in {possible_values}")
#     return [k == e for e in possible_values]


# def one_of_k_encoding_unk(x, allowable_set):
#     if x not in allowable_set:
#         x = allowable_set[-1]
#     return list(map(lambda s: x == s, allowable_set))


# def atom_features(mol, graph, atom_symbols=['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'I'], explicit_H=True):

#     for atom in mol.GetAtoms():
#         results = one_of_k_encoding_unk(atom.GetSymbol(), atom_symbols + ['Unknown']) + \
#                 one_of_k_encoding_unk(atom.GetDegree(),[0, 1, 2, 3, 4, 5, 6]) + \
#                 one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6]) + \
#                 one_of_k_encoding_unk(atom.GetHybridization(), [
#                     Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
#                     Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.
#                                         SP3D, Chem.rdchem.HybridizationType.SP3D2
#                     ]) + [atom.GetIsAromatic()]
#         # In case of explicit hydrogen(QM8, QM9), avoid calling `GetTotalNumHs`
#         if explicit_H:
#             results = results + one_of_k_encoding_unk(atom.GetTotalNumHs(),
#                                                     [0, 1, 2, 3, 4])

#         atom_feats = np.array(results).astype(np.float32)

#         graph.add_node(atom.GetIdx(), feats=torch.from_numpy(atom_feats))

# def get_edge_index(mol, graph):
#     for bond in mol.GetBonds():
#         i = bond.GetBeginAtomIdx()
#         j = bond.GetEndAtomIdx()

#         graph.add_edge(i, j)

# def mol2graph(mol):
#     graph = nx.Graph()
#     atom_features(mol, graph)
#     get_edge_index(mol, graph)

#     graph = graph.to_directed()
#     x = torch.stack([feats['feats'] for n, feats in graph.nodes(data=True)])
#     edge_index = torch.stack([torch.LongTensor((u, v)) for u, v in graph.edges(data=False)]).T

#     return x, edge_index

# def inter_graph(ligand, pocket, dis_threshold = 5.):
#     atom_num_l = ligand.GetNumAtoms()
#     atom_num_p = pocket.GetNumAtoms()

#     graph_inter = nx.Graph()
#     pos_l = ligand.GetConformers()[0].GetPositions()
#     pos_p = pocket.GetConformers()[0].GetPositions()
#     dis_matrix = distance_matrix(pos_l, pos_p)
#     node_idx = np.where(dis_matrix < dis_threshold)
#     for i, j in zip(node_idx[0], node_idx[1]):
#         graph_inter.add_edge(i, j+atom_num_l) 

#     graph_inter = graph_inter.to_directed()
#     edge_index_inter = torch.stack([torch.LongTensor((u, v)) for u, v in graph_inter.edges(data=False)]).T

#     return edge_index_inter

# # %%
# def mols2graphs(complex_path, label, save_path, dis_threshold=5.):
#     try:
#         # Check if complex file exists
#         if not os.path.isfile(complex_path):
#             return

#         with open(complex_path, 'rb') as f:
#             ligand, pocket = pickle.load(f)

#         # Check ligand and pocket validity
#         if ligand is None or pocket is None:
#             print(f"[SKIPPED] Invalid ligand or pocket in: {complex_path}")
#             return

#         atom_num_l = ligand.GetNumAtoms()
#         atom_num_p = pocket.GetNumAtoms()

#         pos_l = torch.FloatTensor(ligand.GetConformers()[0].GetPositions())
#         pos_p = torch.FloatTensor(pocket.GetConformers()[0].GetPositions())
#         x_l, edge_index_l = mol2graph(ligand)
#         x_p, edge_index_p = mol2graph(pocket)
#         x = torch.cat([x_l, x_p], dim=0)
#         edge_index_intra = torch.cat([edge_index_l, edge_index_p + atom_num_l], dim=-1)
#         edge_index_inter = inter_graph(ligand, pocket, dis_threshold=dis_threshold)
#         y = torch.FloatTensor([label])
#         pos = torch.concat([pos_l, pos_p], dim=0)
#         split = torch.cat([torch.zeros((atom_num_l,)), torch.ones((atom_num_p,))], dim=0)

#         data = Data(x=x, edge_index_intra=edge_index_intra, edge_index_inter=edge_index_inter, y=y, pos=pos, split=split)

#         torch.save(data, save_path)
#     except Exception as e:
#         print(f"[ERROR] Failed to process {complex_path}: {e}")
#     # return data

# # %%
# class PLIDataLoader(DataLoader):
#     def __init__(self, data, **kwargs):
#         super().__init__(data, collate_fn=data.collate_fn, **kwargs)

# class GraphDataset(Dataset):
#     """
#     This class is used for generating graph objects using multi process
#     """
#     def __init__(self, data_dir, data_df, dis_threshold=5, graph_type='Graph_GIGN', num_process=8, create=False):
#         self.data_dir = data_dir
#         self.data_df = data_df
#         self.dis_threshold = dis_threshold
#         self.graph_type = graph_type
#         self.create = create
#         self.graph_paths = None
#         self.complex_ids = None
#         self.num_process = num_process
#         self._pre_process()

#     def _pre_process(self):
#         data_dir = self.data_dir
#         data_df = self.data_df
#         graph_type = self.graph_type
#         dis_thresholds = repeat(self.dis_threshold, len(data_df))

#         complex_path_list = []
#         complex_id_list = []
#         pKa_list = []
#         graph_path_list = []
#         for i, row in tqdm(data_df.iterrows(), total=len(data_df), desc="Processing complexes"):
#             if row['pdbid'] == 'index' or row['pdbid'] == 'readme':
#                 continue
#             try:
#                 cid, pKa = row['pdbid'], float(row['-logkd/ki'])
#                 complex_dir = os.path.join(data_dir, cid)
#                 graph_path = os.path.join(complex_dir, f"{graph_type}-{cid}_{self.dis_threshold}A.pyg")
#                 complex_path = os.path.join(complex_dir, f"{cid}_{self.dis_threshold}A.rdkit")

#                 complex_path_list.append(complex_path)
#                 complex_id_list.append(cid)
#                 pKa_list.append(pKa)
#                 graph_path_list.append(graph_path)
#             except Exception as e:
#                 print(f"Error processing {cid}: {e}")
#                 continue

#         if self.create:
#             print('Generate complex graph...')
#             # multi-thread processing
#             pool = multiprocessing.Pool(self.num_process)
#             pool.starmap(mols2graphs,
#                             zip(complex_path_list, pKa_list, graph_path_list, dis_thresholds))
#             pool.close()
#             pool.join()

#         self.graph_paths = graph_path_list
#         self.complex_ids = complex_id_list

#     def __getitem__(self, idx):
#         return torch.load(self.graph_paths[idx])

#     def collate_fn(self, batch):
#         return Batch.from_data_list(batch)

#     def __len__(self):
#         return len(self.data_df)

# if __name__ == '__main__':
#     data_root = './data'
#     toy_dir = os.path.join(data_root, 'PDBbind_v2020_other_PL/v2020-other-PL')
#     toy_df = pd.read_csv(os.path.join(data_root, 'binding_data.csv'))
#     toy_set = GraphDataset(toy_dir, toy_df, graph_type='Graph_GIGN', dis_threshold=5, create=True)
#     train_loader = PLIDataLoader(toy_set, batch_size=128, shuffle=True, num_workers=4)

# # %%

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
from torch.utils.data import Dataset, DataLoader
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


def atom_features(mol, graph, atom_symbols=['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'I'], explicit_H=True):

    for atom in mol.GetAtoms():
        results = one_of_k_encoding_unk(atom.GetSymbol(), atom_symbols + ['Unknown']) + \
                one_of_k_encoding_unk(atom.GetDegree(),[0, 1, 2, 3, 4, 5, 6]) + \
                one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6]) + \
                one_of_k_encoding_unk(atom.GetHybridization(), [
                    Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                    Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.
                                        SP3D, Chem.rdchem.HybridizationType.SP3D2
                    ]) + [atom.GetIsAromatic()]
        # In case of explicit hydrogen(QM8, QM9), avoid calling `GetTotalNumHs`
        if explicit_H:
            results = results + one_of_k_encoding_unk(atom.GetTotalNumHs(),
                                                    [0, 1, 2, 3, 4])

        atom_feats = np.array(results).astype(np.float32)

        graph.add_node(atom.GetIdx(), feats=torch.from_numpy(atom_feats))

def get_edge_index(mol, graph):
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        graph.add_edge(i, j)

def mol2graph(mol):
    graph = nx.Graph()
    atom_features(mol, graph)
    get_edge_index(mol, graph)

    graph = graph.to_directed() # Ensure edges for both directions if model expects it
    x = torch.stack([feats['feats'] for n, feats in graph.nodes(data=True)])
    
    # Check if graph has edges before creating edge_index
    if graph.edges():
        edge_index = torch.stack([torch.LongTensor((u, v)) for u, v in graph.edges(data=False)]).T
    else:
        edge_index = torch.empty((2,0), dtype=torch.long)


    return x, edge_index

def inter_graph(ligand, pocket, dis_threshold = 5.):
    atom_num_l = ligand.GetNumAtoms()
    # atom_num_p = pocket.GetNumAtoms() # Not directly used here

    graph_inter = nx.Graph()
    pos_l = ligand.GetConformers()[0].GetPositions()
    pos_p = pocket.GetConformers()[0].GetPositions()
    dis_matrix = distance_matrix(pos_l, pos_p)
    node_idx = np.where(dis_matrix < dis_threshold)
    for i, j in zip(node_idx[0], node_idx[1]):
        graph_inter.add_edge(i, j+atom_num_l) 

    graph_inter = graph_inter.to_directed() # Ensure edges for both directions
    
    if graph_inter.edges():
        edge_index_inter = torch.stack([torch.LongTensor((u, v)) for u, v in graph_inter.edges(data=False)]).T
    else:
        edge_index_inter = torch.empty((2,0), dtype=torch.long)

    return edge_index_inter

# %%
def mols2graphs(complex_path, label, save_path, dis_threshold=5.):
    try:
        # Check if complex file exists
        if not os.path.isfile(complex_path):
            # print(f"[SKIPPING] Complex file not found: {complex_path}")
            return

        with open(complex_path, 'rb') as f:
            ligand, pocket = pickle.load(f)

        # Check ligand and pocket validity
        if ligand is None or pocket is None:
            print(f"[SKIPPED] Invalid ligand or pocket in: {complex_path}")
            return
        
        # Ensure conformers exist
        if ligand.GetNumConformers() == 0:
            print(f"[SKIPPED] Ligand in {complex_path} has no conformers.")
            return
        if pocket.GetNumConformers() == 0:
            print(f"[SKIPPED] Pocket in {complex_path} has no conformers.")
            return

        atom_num_l = ligand.GetNumAtoms()
        atom_num_p = pocket.GetNumAtoms()

        if atom_num_l == 0 :
            print(f"[SKIPPED] Ligand in {complex_path} has no atoms.")
            return
        # Pocket can sometimes be empty if no atoms are within threshold, but usually not desirable
        # if atom_num_p == 0:
        #     print(f"[SKIPPING] Pocket in {complex_path} has no atoms.")
        #     return


        pos_l = torch.FloatTensor(ligand.GetConformers()[0].GetPositions())
        pos_p = torch.FloatTensor(pocket.GetConformers()[0].GetPositions())
        
        x_l, edge_index_l = mol2graph(ligand)
        x_p, edge_index_p = mol2graph(pocket)

        x = torch.cat([x_l, x_p], dim=0) if atom_num_p > 0 else x_l
        
        edge_index_p_shifted = edge_index_p + atom_num_l if atom_num_p > 0 and edge_index_p.numel() > 0 else torch.empty((2,0), dtype=torch.long)
        
        edge_index_intra = torch.cat([edge_index_l, edge_index_p_shifted], dim=-1)
        edge_index_inter = inter_graph(ligand, pocket, dis_threshold=dis_threshold)
        
        # Combined edge_index for general GNN layers (e.g. non-GIGN SE)
        # Ensure both are 2xN before concat, handle empty tensors
        list_to_cat = []
        if edge_index_intra.numel() > 0:
            list_to_cat.append(edge_index_intra)
        if edge_index_inter.numel() > 0:
            list_to_cat.append(edge_index_inter)
        
        if list_to_cat:
            edge_index = torch.cat(list_to_cat, dim=-1)
        else: # Case where both intra and inter might be empty (e.g. single atom ligand, no pocket atoms/interaction)
            edge_index = torch.empty((2,0), dtype=torch.long)


        y = torch.FloatTensor([label])
        pos = torch.cat([pos_l, pos_p], dim=0) if atom_num_p > 0 else pos_l
        split = torch.cat([torch.zeros((atom_num_l,)), torch.ones((atom_num_p,))], dim=0) if atom_num_p > 0 else torch.zeros((atom_num_l,))

        data = Data(x=x, edge_index=edge_index, # Combined edge index
                    edge_index_intra=edge_index_intra, 
                    edge_index_inter=edge_index_inter, 
                    y=y, pos=pos, split=split)

        torch.save(data, save_path)
    except Exception as e:
        print(f"[ERROR] Failed to process {complex_path}: {e}")
    # return data

# %%
class PLIDataLoader(DataLoader):
    def __init__(self, data, **kwargs):
        super().__init__(data, collate_fn=data.collate_fn, **kwargs)

class GraphDataset(Dataset):
    """
    This class is used for generating graph objects using multi process.
    It can be adapted to fit the PyG Dataset standard if needed, by saving
    processed files to `self.processed_dir` and implementing `len` and `get`.
    For now, it uses its own path management.
    """
    def __init__(self, data_dir, data_df, dis_threshold=5, graph_type='Graph_GIGN', num_process=8, create=False):
        self.data_dir = data_dir # This should be the root of PDBbind specific folders, e.g., v2020-other-PL
        self.data_df = data_df
        self.dis_threshold = dis_threshold
        self.graph_type = graph_type # Used in filename for .pyg files
        self.create = create
        self.graph_paths = [] # List of paths to .pyg files
        self.complex_ids = [] # Corresponding complex_ids
        self.num_process = num_process
        self._pre_process()

    def _pre_process(self):
        data_dir = self.data_dir 
        data_df = self.data_df
        graph_type = self.graph_type
        
        # Prepare arguments for mols2graphs
        complex_paths_to_process = []
        pKa_values = []
        graph_save_paths = []
        dis_thresholds_list = [] # For starmap

        print("Preparing file list for processing...")
        # First pass: collect all valid paths and data
        temp_graph_paths = [] # Store all potential graph paths
        for i, row in tqdm(data_df.iterrows(), total=len(data_df), desc="Scanning CSV for complexes"):
            if row['pdbid'] == 'index' or row['pdbid'] == 'readme':
                continue
            try:
                cid = str(row['pdbid'])
                pKa = float(row['-logkd/ki'])
                
                complex_specific_dir = os.path.join(data_dir, cid) # e.g., ./data/PDBbind_v2020_other_PL/v2020-other-PL/1a0q
                
                # Path to the .rdkit file (input for mols2graphs)
                # This assumes preprocess.py (generate_complex) has run and created these .rdkit files
                rdkit_complex_path = os.path.join(complex_specific_dir, f"{cid}_{self.dis_threshold}A.rdkit")

                # Path where the .pyg graph file will be saved by mols2graphs
                graph_pyg_path = os.path.join(complex_specific_dir, f"{graph_type}-{cid}_{self.dis_threshold}A.pyg")
                
                temp_graph_paths.append(graph_pyg_path) # Store for later loading

                if self.create:
                    # If creating, check if .rdkit file exists. If not, it cannot be processed.
                    if not os.path.exists(rdkit_complex_path):
                        # print(f"RDKit file not found for {cid}, skipping generation: {rdkit_complex_path}")
                        continue
                    
                    # Add to list for processing only if not already created or forced update
                    # For simplicity, let's assume we always try to process if create=True and .rdkit exists.
                    # A check for existing .pyg could be added here to skip already processed ones.
                    # if os.path.exists(graph_pyg_path): # Optional: skip if .pyg already exists
                    #    continue

                    complex_paths_to_process.append(rdkit_complex_path)
                    pKa_values.append(pKa)
                    graph_save_paths.append(graph_pyg_path)
                    dis_thresholds_list.append(self.dis_threshold)
                
                # Regardless of self.create, we populate self.graph_paths if the .pyg file exists or will be created
                # This means if create=False, we only consider existing .pyg files.
                # If create=True, we consider all .pyg files that will be (or are) generated.

            except Exception as e:
                print(f"Error preparing data for {row.get('pdbid', 'Unknown PDBID')}: {e}")
                continue
        
        if self.create and complex_paths_to_process:
            print(f"Generating {len(complex_paths_to_process)} complex graphs...")
            # Multi-thread processing
            with multiprocessing.Pool(self.num_process) as pool:
                pool.starmap(mols2graphs,
                             zip(complex_paths_to_process, pKa_values, graph_save_paths, dis_thresholds_list))
            print("Graph generation complete.")

        # Second pass: Populate self.graph_paths with existing .pyg files
        # This ensures that only successfully created/existing .pyg files are part of the dataset
        print("Verifying existing graph files...")
        final_graph_paths = []
        final_complex_ids = [] # Store corresponding PDB IDs
        # Iterate through the original dataframe again to maintain order and link PDB IDs
        for i, row in tqdm(data_df.iterrows(), total=len(data_df), desc="Finalizing graph list"):
            if row['pdbid'] == 'index' or row['pdbid'] == 'readme':
                continue
            try:
                cid = str(row['pdbid'])
                complex_specific_dir = os.path.join(data_dir, cid)
                graph_pyg_path = os.path.join(complex_specific_dir, f"{graph_type}-{cid}_{self.dis_threshold}A.pyg")
                if os.path.exists(graph_pyg_path):
                    # Check if graph can be loaded (basic integrity check)
                    try:
                        _ = torch.load(graph_pyg_path) # Try loading
                        final_graph_paths.append(graph_pyg_path)
                        final_complex_ids.append(cid)
                    except Exception as load_e:
                        print(f"Could not load .pyg file {graph_pyg_path}, skipping: {load_e}")
                # else:
                    # print(f".pyg file not found after processing for {cid}, skipping: {graph_pyg_path}")
            except Exception as e:
                print(f"Error finalizing graph path for {row.get('pdbid', 'Unknown PDBID')}: {e}")
                continue

        self.graph_paths = final_graph_paths
        self.complex_ids = final_complex_ids # Store PDB IDs if needed later
        print(f"Dataset initialized with {len(self.graph_paths)} graphs.")


    def __getitem__(self, idx):
        # This might return None if the file is corrupted or cannot be loaded
        try:
            data = torch.load(self.graph_paths[idx])
            # Basic check for essential attributes
            if not all(hasattr(data, attr) for attr in ['x', 'edge_index_intra', 'edge_index_inter', 'pos', 'y']):
                 print(f"Warning: Data at {self.graph_paths[idx]} is missing essential attributes. Returning None.")
                 return None
            if data.x is None or data.x.nelement() == 0:
                 print(f"Warning: Data at {self.graph_paths[idx]} has no node features (x is None or empty). Returning None.")
                 return None
            if data.y is None:
                print(f"Warning: Data at {self.graph_paths[idx]} has no target (y is None). Returning None.")
                return None

            # Add pdb_id to data object if you need it (e.g., for filtering in train script)
            if self.complex_ids: # Check if complex_ids were populated
                 data.pdb_id = self.complex_ids[idx]

            return data
        except Exception as e:
            print(f"Error loading graph {self.graph_paths[idx]} at index {idx}: {e}. Returning None.")
            return None


    def collate_fn(self, batch):
        # Filter out None items from the batch, which may occur if __getitem__ returns None
        batch = [item for item in batch if item is not None]
        if not batch:
            return None # Return None if the entire batch is invalid
        return Batch.from_data_list(batch)

    def __len__(self):
        return len(self.graph_paths)

if __name__ == '__main__':
    # This main block is for testing dataset_GIGN.py
    # Ensure your paths are correct for your environment
    data_root_dir = './data'  # This should be the directory containing 'PDBbind_v2020_other_PL' and 'binding_data.csv'
    
    # Path to the directory containing individual complex folders (e.g., '1a0q', '1a0s')
    # This is what 'data_dir' in GraphDataset expects.
    complex_files_dir = os.path.join(data_root_dir, 'PDBbind_v2020_other_PL/v2020-other-PL')
    
    # Path to the CSV file with PDB IDs and binding affinities
    binding_data_csv_path = os.path.join(data_root_dir, 'binding_data.csv')

    if not os.path.exists(complex_files_dir):
        print(f"Error: Complex files directory not found: {complex_files_dir}")
        print("Please ensure PDBbind data is downloaded and extracted correctly, and preprocess.py has run to generate .rdkit files.")
    elif not os.path.exists(binding_data_csv_path):
        print(f"Error: Binding data CSV not found: {binding_data_csv_path}")
        print("Please ensure create_binding_ds.py has run.")
    else:
        print(f"Using complex files directory: {complex_files_dir}")
        print(f"Using binding data CSV: {binding_data_csv_path}")

        toy_df = pd.read_csv(binding_data_csv_path)
        
        # Example: Create a small subset for testing
        # toy_df = toy_df.head(20) # Take first 20 entries for quick test
        
        print(f"DataFrame shape: {toy_df.shape}")
        if toy_df.empty:
            print("Error: DataFrame is empty. Check CSV content and path.")
        else:
            # Set create=True to generate .pyg files.
            # This assumes preprocess.py was run to create .rdkit files.
            # If .pyg files already exist and create=False, it will just load them.
            toy_set = GraphDataset(data_dir=complex_files_dir, 
                                   data_df=toy_df, 
                                   graph_type='Graph_GIGN_Test', # Use a distinct graph_type for testing
                                   dis_threshold=5, 
                                   create=True, # Set to True to generate graphs
                                   num_process=4) # Adjust num_process as needed
            
            print(f"Number of graphs in dataset: {len(toy_set)}")

            if len(toy_set) > 0:
                # Test DataLoader
                # Use collate_fn from the dataset instance
                train_loader = PLIDataLoader(toy_set, batch_size=4, shuffle=True, num_workers=0) # num_workers=0 for easier debugging
                
                print(f"Number of batches in DataLoader: {len(train_loader)}")
                
                # Iterate over a few batches to test
                for i, batch_data in enumerate(train_loader):
                    if i >= 3: # Print info for first 3 batches
                        break
                    if batch_data is None:
                        print(f"Batch {i+1} is None (all items were invalid).")
                        continue
                    print(f"Batch {i+1}:")
                    print(f"  Type: {type(batch_data)}")
                    print(f"  Number of graphs: {batch_data.num_graphs}")
                    print(f"  Node features shape: {batch_data.x.shape}")
                    print(f"  Edge index (combined) shape: {batch_data.edge_index.shape}")
                    print(f"  Edge index intra shape: {batch_data.edge_index_intra.shape}")
                    print(f"  Edge index inter shape: {batch_data.edge_index_inter.shape}")
                    print(f"  Positions shape: {batch_data.pos.shape}")
                    print(f"  Target y shape: {batch_data.y.shape}")
                    if hasattr(batch_data, 'pdb_id'): # If you added pdb_id
                         print(f"  PDB IDs in batch: {batch_data.pdb_id}")
                
                # Test __getitem__ for a few items
                for i in range(min(5, len(toy_set))):
                    item = toy_set[i]
                    if item:
                        print(f"Item {i}: {item}")
                    else:
                        print(f"Item {i} is None.")
            else:
                print("Dataset is empty. Cannot test DataLoader or __getitem__.")
                print("Possible reasons: ")
                print("1. `create=True` but .rdkit files (from preprocess.py) are missing for all entries in the CSV.")
                print("2. `create=False` and no .pyg files were found matching the criteria.")
                print("3. Errors during graph generation for all items.")
                print("4. The input CSV (`binding_data.csv`) might be empty or only contain 'index'/'readme' rows.")


# %%
