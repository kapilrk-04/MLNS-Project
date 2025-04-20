import torch
import torch.nn as nn
import torch.optim as optim
# from torch_geometric.loader import DataLoader # No, use PLIDataLoader from dataset_GIGN
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import argparse
import os
import time
import random
import pandas as pd # For reading CSV

# Assuming all files are in the same directory:
from dataset_GIGN import GraphDataset as PDBBindDataset # Alias for convenience
from dataset_GIGN import PLIDataLoader # Use the DataLoader from dataset_GIGN
from regression_model import GraphTransformerRegression

# Helper function for counting parameters (usually in utils.py)
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# collate_fn is now a static method in PDBBindDataset (GraphDataset in dataset_GIGN.py)
# So, we can pass it directly to DataLoader if not using PyGDataLoader's default.
# PLIDataLoader in dataset_GIGN.py is already PyGDataLoader, handles it.

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    valid_graphs_count = 0 
    start_time = time.time()
    
    for batch_data in loader:
        if batch_data is None: continue # Skip if collate_fn returned None

        batch_data = batch_data.to(device)
        
        # Validate batch_data (already done more extensively in dataset __getitem__)
        if not hasattr(batch_data, 'y') or batch_data.y is None or \
           not hasattr(batch_data, 'x') or batch_data.x is None or batch_data.x.shape[0] == 0:
            # print("Skipping batch due to missing essential data (x, y).")
            continue
        if batch_data.batch is None : # Should be provided by PyG Dataloader
            # print("Skipping batch due to missing batch attribute.")
            continue

        targets = batch_data.y.float()
        if targets.ndim > 1 and targets.shape[1] == 1:
            targets = targets.squeeze(1)
        if targets.shape[0] != batch_data.num_graphs:
            # print(f"Target shape {targets.shape} mismatch with num_graphs {batch_data.num_graphs}. Skipping batch.")
            continue

        optimizer.zero_grad()
        try:
            outputs = model(batch_data)
            if outputs.shape != targets.shape:
                # print(f"Output shape {outputs.shape} mismatch with target shape {targets.shape}. Skipping batch.")
                continue
            
            loss = criterion(outputs, targets)
            if torch.isnan(loss):
                # print("NaN loss detected. Skipping backward pass.")
                continue
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch_data.num_graphs
            valid_graphs_count += batch_data.num_graphs
        except Exception as e:
            # print(f"Error during training batch: {e}")
            # import traceback
            # traceback.print_exc() # For detailed error
            continue # Skip batch on error

    epoch_time = time.time() - start_time
    avg_loss = total_loss / valid_graphs_count if valid_graphs_count > 0 else 0
    return avg_loss, epoch_time


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds, all_targets = [], []
    valid_samples_count = 0

    with torch.no_grad():
        for batch_data in loader:
            if batch_data is None: continue
            batch_data = batch_data.to(device)

            if not hasattr(batch_data, 'y') or batch_data.y is None or \
               not hasattr(batch_data, 'x') or batch_data.x is None or batch_data.x.shape[0] == 0:
                continue
            if batch_data.batch is None : continue


            targets = batch_data.y.float()
            if targets.ndim > 1 and targets.shape[1] == 1: targets = targets.squeeze(1)
            if targets.shape[0] != batch_data.num_graphs: continue
            
            try:
                outputs = model(batch_data)
                if outputs.shape != targets.shape: continue

                loss = criterion(outputs, targets)
                if torch.isnan(loss): continue

                total_loss += loss.item() * batch_data.num_graphs
                valid_samples_count += batch_data.num_graphs
                all_preds.append(outputs.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
            except Exception as e:
                # print(f"Error during evaluation batch: {e}")
                continue
    
    avg_loss = total_loss / valid_samples_count if valid_samples_count > 0 else 0
    if not all_preds or not all_targets: return avg_loss, np.nan, np.nan

    all_preds_np = np.concatenate(all_preds).flatten()
    all_targets_np = np.concatenate(all_targets).flatten()

    if len(all_targets_np) == 0: return avg_loss, np.nan, np.nan
    
    mse = mean_squared_error(all_targets_np, all_preds_np)
    mae = mean_absolute_error(all_targets_np, all_preds_np)
    return avg_loss, np.sqrt(mse), mae


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GIGN-Transformer for Binding Affinity")
    # Data args
    parser.add_argument("--data_root", type=str, default="./data", help="Root directory (e.g., './data') containing PDBbind dir and binding_data.csv")
    parser.add_argument("--pdbbind_dir_name", type=str, default="PDBbind_v2020_other_PL/v2020-other-PL", help="Name of the PDBbind directory relative to data_root.")
    parser.add_argument("--binding_csv_name", type=str, default="binding_data.csv", help="Name of the binding data CSV relative to data_root.")
    parser.add_argument("--graph_type_dataset", type=str, default="GIGN_GraphTransformer_Example", help="Prefix for processed graph filenames by dataset_GIGN.")
    parser.add_argument("--create_dataset_graphs", action="store_true", help="Force (re)generation of .pyg graph files by dataset_GIGN.")
    parser.add_argument("--num_process_dataset", type=int, default=4, help="Number of processes for dataset_GIGN graph creation.")
    parser.add_argument("--add_dummy_edge_attr", action="store_true", help="Add dummy edge attributes in dataset_GIGN (for testing use_edge_attr in model).")

    # Model HPs
    parser.add_argument("--d_model", type=int, default=64, help="Model dimension.") # Smaller for example
    parser.add_argument("--num_heads", type=int, default=4, help="Attention heads.")
    parser.add_argument("--dim_feedforward", type=int, default=128, help="FFN dimension.")
    parser.add_argument("--num_layers", type=int, default=2, help="Transformer layers.") # Fewer for example
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate.")
    parser.add_argument("--se", type=str, default="gign", choices=["gign", "gnn", "khopgnn"], help="Structure extractor type.")
    parser.add_argument("--gnn_type_se", type=str, default="gcn", help="GNN type if se is 'gnn' or 'khopgnn'.")
    parser.add_argument("--k_hop_se", type=int, default=2, help="K-hop for GNN/KHop SE.")
    parser.add_argument("--num_gign_layers", type=int, default=2, help="Number of HIL layers in GIGN SE.")
    parser.add_argument("--no_use_edge_attr", action="store_false", dest="use_edge_attr", help="Do NOT use edge attributes in model (even if dataset provides them).")
    parser.set_defaults(use_edge_attr=True) # Default to True, depends on --add_dummy_edge_attr for dataset
    parser.add_argument("--global_pool", type=str, default="mean", choices=['mean', 'add', 'cls'], help="Global pooling.")
    parser.add_argument("--batch_norm", action="store_true", help="Use BatchNorm in Transformer layers.")
    
    # Training HPs
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs.") # Short for example
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size.") # Smaller for example
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay.")
    parser.add_argument("--test_split", type=float, default=0.1)
    parser.add_argument("--val_split", type=float, default=0.1) # of remaining
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers_loader", type=int, default=0, help="DataLoader workers.")
    parser.add_argument("--max_complexes", type=int, default=50, help="Max complexes to load for quick test (0 for all).") # For quick run
    parser.add_argument("--output_dir", type=str, default="output_gign_transformer_example")
    args = parser.parse_args()

    # Setup
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    log_file_path = os.path.join(args.output_dir, "training_log.txt")

    print(f"Using device: {device}")
    print(f"Output directory: {args.output_dir}")
    print(f"Log file: {log_file_path}")

    # Data
    binding_csv_full_path = os.path.join(args.data_root, args.binding_csv_name)
    pdbbind_data_full_path = os.path.join(args.data_root, args.pdbbind_dir_name)

    if not os.path.exists(binding_csv_full_path):
        raise FileNotFoundError(f"Binding CSV not found: {binding_csv_full_path}")
    if not os.path.exists(pdbbind_data_full_path):
        raise FileNotFoundError(f"PDBbind data directory not found: {pdbbind_data_full_path}")

    print("Loading dataset CSV...")
    full_df = pd.read_csv(binding_csv_full_path)
    if args.max_complexes > 0:
        print(f"Using a subset of {args.max_complexes} complexes for this example.")
        data_df = full_df.head(args.max_complexes)
    else:
        data_df = full_df
    
    print(f"Initializing PDBBindDataset (dataset_GIGN.GraphDataset) with {len(data_df)} complexes...")
    # Note: PDBBindDataset is an alias for dataset_GIGN.GraphDataset
    # It expects data_dir to be the path to the directory containing individual complex folders (e.g., .../v2020-other-PL/)
    full_dataset = PDBBindDataset(
        data_dir=pdbbind_data_full_path,
        data_df=data_df,
        graph_type=args.graph_type_dataset,
        create=args.create_dataset_graphs,
        num_process=args.num_process_dataset,
        add_dummy_edge_attr=args.add_dummy_edge_attr # Pass this flag
    )

    if len(full_dataset) == 0:
        raise ValueError("Dataset is empty. Check paths, .rdkit files, and dataset creation flags/logs.")

    # Determine feature sizes from the first valid sample
    first_valid_sample = None
    for i in range(len(full_dataset)):
        sample = full_dataset[i]
        if sample is not None:
            first_valid_sample = sample
            break
    if first_valid_sample is None:
        raise ValueError("Could not retrieve a valid sample from the dataset to determine feature sizes.")
    
    in_node_size = first_valid_sample.x.shape[1]
    # Edge attributes handling:
    # If args.use_edge_attr is True (model wants to use them) AND
    # args.add_dummy_edge_attr is True (dataset is providing them), then num_edge_features is from sample.
    # Otherwise, num_edge_features is 0 and model's use_edge_attr should effectively be False.
    actual_use_edge_attr_model = args.use_edge_attr and args.add_dummy_edge_attr
    num_edge_features = 0
    if actual_use_edge_attr_model:
        if hasattr(first_valid_sample, 'edge_attr') and first_valid_sample.edge_attr is not None:
            num_edge_features = first_valid_sample.edge_attr.shape[1]
            print(f"Dummy edge attributes enabled in dataset. Model will use them. Edge feature dim: {num_edge_features}")
        else: # Dataset was supposed to add them but didn't, or first sample has none.
            print("Warning: Model expects edge_attr, but dataset (or first sample) doesn't provide them. Disabling edge_attr in model.")
            actual_use_edge_attr_model = False
    else:
        print("Edge attributes are disabled for the model (either by --no_use_edge_attr or --add_dummy_edge_attr=False).")


    print(f"Input node feature size: {in_node_size}")
    print(f"Model use_edge_attr: {actual_use_edge_attr_model}, Num edge features: {num_edge_features}")


    # Train/Val/Test Split
    dataset_size = len(full_dataset)
    indices = list(range(dataset_size))
    random.shuffle(indices) # Shuffle before splitting

    num_test = int(dataset_size * args.test_split)
    num_val = int((dataset_size - num_test) * args.val_split)
    
    test_indices = indices[:num_test]
    val_indices = indices[num_test : num_test + num_val]
    train_indices = indices[num_test + num_val :]

    # PyG's Dataset.index_select creates a new dataset with only the selected indices
    train_dataset = full_dataset.index_select(train_indices)
    val_dataset = full_dataset.index_select(val_indices) if val_indices else None
    test_dataset = full_dataset.index_select(test_indices) if test_indices else None
    
    print(f"Dataset split: Train={len(train_dataset)}, Val={len(val_dataset) if val_dataset else 0}, Test={len(test_dataset) if test_dataset else 0}")

    if len(train_dataset) == 0: raise ValueError("Training set is empty!")

    # DataLoaders (using PLIDataLoader from dataset_GIGN which is PyGDataLoader)
    # Pass the dataset's static collate_fn if using PyGDataLoader and it's needed (it filters Nones)
    train_loader = PLIDataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                                 num_workers=args.num_workers_loader, collate_fn=PDBBindDataset.collate_fn)
    val_loader = PLIDataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, 
                               num_workers=args.num_workers_loader, collate_fn=PDBBindDataset.collate_fn) if val_dataset else None
    test_loader = PLIDataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, 
                                num_workers=args.num_workers_loader, collate_fn=PDBBindDataset.collate_fn) if test_dataset else None

    # Model
    model_kwargs = {
        'k_hop': args.k_hop_se,
        'num_gign_layers': args.num_gign_layers,
        # 'deg': pna_degrees_tensor # If PNA is used and 'deg' is precomputed for the batch
    }
    model = GraphTransformerRegression(
        in_size=in_node_size, out_dim=1, d_model=args.d_model, num_heads=args.num_heads,
        dim_feedforward=args.dim_feedforward, dropout=args.dropout, num_layers=args.num_layers,
        batch_norm=args.batch_norm, se=args.se, gnn_type=args.gnn_type_se,
        use_edge_attr=actual_use_edge_attr_model, # Use the determined value
        num_edge_features=num_edge_features,    # Use the determined value
        global_pool=args.global_pool,
        **model_kwargs
    ).to(device)
    print(f"Model initialized: {args.se} SE. Parameters: {count_parameters(model):,}")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True) # shorter patience for example

    # Training Loop
    print("\nStarting training...")
    best_val_rmse = float('inf')
    
    with open(log_file_path, 'w') as f_log:
        header = "Epoch,Train_Loss,Val_Loss,Val_RMSE,Val_MAE,Time_s,LR\n"
        f_log.write(header)
        print(header.strip())

        for epoch in range(1, args.epochs + 1):
            train_loss, epoch_time = train_epoch(model, train_loader, criterion, optimizer, device)
            
            val_loss, val_rmse, val_mae = float('nan'), float('nan'), float('nan')
            if val_loader:
                val_loss, val_rmse, val_mae = evaluate(model, val_loader, criterion, device)
                if not np.isnan(val_rmse): scheduler.step(val_rmse)
                elif not np.isnan(val_loss): scheduler.step(val_loss)


            current_lr = optimizer.param_groups[0]['lr']
            log_entry = f"{epoch},{train_loss:.4f},{val_loss:.4f},{val_rmse:.4f},{val_mae:.4f},{epoch_time:.2f},{current_lr:.2e}"
            print(log_entry)
            f_log.write(log_entry + "\n")
            f_log.flush()

            if val_loader and not np.isnan(val_rmse) and val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                torch.save(model.state_dict(), os.path.join(args.output_dir, "best_model.pt"))
                # print(f"Epoch {epoch}: New best Val RMSE: {best_val_rmse:.4f}. Model saved.")

    print("\nTraining finished.")
    if val_loader : print(f"Best Validation RMSE: {best_val_rmse:.4f}")

    # Final Test
    if test_loader:
        print("\nEvaluating on Test Set...")
        best_model_path = os.path.join(args.output_dir, "best_model.pt")
        if os.path.exists(best_model_path) and val_loader: # Only load best if val was done
            try:
                model.load_state_dict(torch.load(best_model_path, map_location=device))
                print("Loaded best model for test evaluation.")
            except Exception as e:
                print(f"Error loading best model: {e}. Testing with final model state.")
        else:
            print("Testing with final model state (no best model from validation or no val set).")
        
        test_loss, test_rmse, test_mae = evaluate(model, test_loader, criterion, device)
        test_log = f"\nTest Results: Loss={test_loss:.4f}, RMSE={test_rmse:.4f}, MAE={test_mae:.4f}"
        print(test_log)
        with open(log_file_path, 'a') as f_log:
            f_log.write(test_log + "\n")
    else:
        print("\nNo test set for final evaluation.")