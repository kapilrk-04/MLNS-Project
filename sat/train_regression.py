import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset # Required for type hints if needed
from torch.utils.data import Subset # Required for type hints if needed
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import argparse
import os
import time
import random

# Import your custom dataset and model
from pdbbind_dataset import PDBBindDataset # Assuming it's in pdbbind_dataset.py
from regression_model import GraphTransformerRegression # Use the regression model

# Import other necessary components from your project structure
# e.g., from metrics import ... (if you have custom regression metrics)
from utils import count_parameters # Assuming count_parameters is in utils.py

# Helper function from preprocess.py to calculate in_size (removed as we get size dynamically now)
# from preprocess import POSSIBLE_ATOM_TYPES, POSSIBLE_HYBRIDIZATION

# def calculate_in_size(): # Replaced with dynamic calculation
#     return len(POSSIBLE_ATOM_TYPES) + len(POSSIBLE_HYBRIDIZATION) + 1 + 1 + 1 + 1 + 1

# Function to handle None data from dataset.get() in DataLoader
def collate_filter_none(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if not batch:
        return None # Return None if the whole batch is invalid
    # Use PyG's default collate if batch is not empty
    # Import locally to avoid potential circular dependencies if Collater is complex
    from torch_geometric.data.dataloader import Collater
    return Collater(follow_batch=[], exclude_keys=[])(batch)


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    valid_graphs = 0 # Use graph count for averaging
    start_time = time.time()
    batch_num = 0

    for batch_data in loader:
        batch_num += 1
        # Handle potentially empty batches from collate_filter_none
        if batch_data is None:
            # print(f"Skipping empty batch {batch_num}.")
            continue

        # Check if batch_data is on the correct device, move if necessary
        if batch_data.batch.device != device:
             batch_data = batch_data.to(device)

        # Ensure target data `y` is available and correctly shaped
        if not hasattr(batch_data, 'y') or batch_data.y is None:
            print(f"Skipping batch {batch_num} due to missing target 'y'.")
            continue

        # Ensure y is float and has the right shape (e.g., [batch_size])
        try:
            targets = batch_data.y.float()
            if targets.ndim > 1 and targets.shape[1] == 1:
                targets = targets.squeeze(1) # Ensure shape [batch_size]
            # Check if target tensor has the correct number of elements
            if targets.shape[0] != batch_data.num_graphs:
                 print(f"Warning: Target shape {targets.shape} mismatch with num_graphs {batch_data.num_graphs} in batch {batch_num}. Skipping.")
                 continue

        except Exception as e:
            print(f"Error processing targets in batch {batch_num}: {e}. Skipping.")
            continue

        optimizer.zero_grad()

        try:
            outputs = model(batch_data) # Get predictions

            # Ensure outputs and targets have compatible shapes
            if outputs.shape != targets.shape:
                 print(f"Shape mismatch in batch {batch_num}: Output {outputs.shape}, Target {targets.shape}. Skipping.")
                 continue

            loss = criterion(outputs, targets)

            if torch.isnan(loss):
                print(f"Warning: NaN loss detected in batch {batch_num}. Skipping backward pass.")
                continue # Skip backward pass for this batch

            loss.backward()
            # Optional: Gradient clipping
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item() * batch_data.num_graphs # Scale loss by number of graphs
            valid_graphs += batch_data.num_graphs

        except Exception as e:
            print(f"Error during model forward/backward in batch {batch_num}: {e}")
            # Consider stopping or just skipping the batch
            continue


    end_time = time.time()
    epoch_time = end_time - start_time
    avg_loss = total_loss / valid_graphs if valid_graphs > 0 else 0
    # print(f"Train Epoch Time: {epoch_time:.2f}s") # Moved timing print to main loop
    return avg_loss, epoch_time

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    valid_samples = 0
    batch_num = 0

    with torch.no_grad():
        for batch_data in loader:
            batch_num += 1
            if batch_data is None: continue # Skip empty batches

            if batch_data.batch.device != device:
                 batch_data = batch_data.to(device)

            if not hasattr(batch_data, 'y') or batch_data.y is None: continue # Skip if no target

            try:
                targets = batch_data.y.float()
                if targets.ndim > 1 and targets.shape[1] == 1:
                    targets = targets.squeeze(1)
                if targets.shape[0] != batch_data.num_graphs: continue # Skip if mismatch

                outputs = model(batch_data)

                if outputs.shape != targets.shape:
                    print(f"Eval Shape mismatch in batch {batch_num}: Output {outputs.shape}, Target {targets.shape}. Skipping.")
                    continue

                loss = criterion(outputs, targets)

                if torch.isnan(loss):
                    print(f"Warning: NaN loss detected during evaluation batch {batch_num}.")
                    continue # Don't include NaN in average loss

                total_loss += loss.item() * batch_data.num_graphs
                valid_samples += batch_data.num_graphs

                all_preds.append(outputs.cpu().numpy())
                all_targets.append(targets.cpu().numpy())

            except Exception as e:
                 print(f"Error during evaluation batch {batch_num}: {e}")
                 continue


    avg_loss = total_loss / valid_samples if valid_samples > 0 else 0
    if not all_preds: # Handle case where evaluation set was empty or all batches were skipped
         print("Warning: No valid predictions generated during evaluation.")
         return avg_loss, np.nan, np.nan # Return NaN for metrics

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    # Ensure targets and predictions are flat for metrics
    all_preds = all_preds.flatten()
    all_targets = all_targets.flatten()

    if len(all_targets) == 0 or len(all_preds) == 0 or len(all_targets) != len(all_preds):
        print("Warning: Target or prediction array empty or mismatched after concatenation during evaluation.")
        return avg_loss, np.nan, np.nan

    mse = mean_squared_error(all_targets, all_preds)
    mae = mean_absolute_error(all_targets, all_preds)
    rmse = np.sqrt(mse)

    return avg_loss, rmse, mae


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Graph Transformer for Binding Affinity Prediction")
    parser.add_argument("processed_data_dir", type=str, help="Root directory containing the 'processed' subdirectory with .pt files.")
    parser.add_argument("--output_dir", type=str, default="output_regression", help="Directory to save model checkpoints and logs.")
    # --- Filtering ---
    parser.add_argument("--max_nodes", type=int, default=10000, help="Maximum number of nodes (atoms) allowed per graph. Set 0 for no limit.")
    # --- Model Hyperparameters ---
    parser.add_argument("--d_model", type=int, default=128, help="Model dimension.")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads.")
    parser.add_argument("--dim_feedforward", type=int, default=512, help="Feedforward layer dimension.")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of Transformer layers.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate.")
    parser.add_argument("--gnn_type", type=str, default="gcn", help="GNN type for structure extractor (e.g., gcn, gin, pna).")
    parser.add_argument("--se", type=str, default="gnn", help="Structure extractor type (gnn, khopgnn).")
    parser.add_argument("--k_hop", type=int, default=2, help="K-hop for SE (layers for gnn, hop for khopgnn).")
    parser.add_argument("--no_edge_attr", action="store_true", help="Do NOT use edge attributes.")
    parser.add_argument("--global_pool", type=str, default="mean", choices=['mean', 'add', 'cls'], help="Global pooling strategy.")
    parser.add_argument("--batch_norm", action="store_true", help="Use BatchNorm instead of LayerNorm.") # Default is False now
    parser.add_argument("--no_batch_norm", action="store_false", dest="batch_norm", help="Disable BatchNorm (use LayerNorm).")
    parser.set_defaults(batch_norm=False) # Set default to False for BatchNorm
    # --- Training Hyperparameters ---
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay for optimizer.")
    parser.add_argument("--test_split", type=float, default=0.1, help="Fraction of data for testing.")
    parser.add_argument("--val_split", type=float, default=0.1, help="Fraction of *remaining* data (after test split) for validation.") # Clarified meaning
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of workers for DataLoader.")
    args = parser.parse_args()

    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        # CUDNN settings (optional, can sometimes improve reproducibility)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Data Loading ---
    print("Loading dataset...")
    # Load the full dataset first
    try:
        full_dataset_raw = PDBBindDataset(root=args.processed_data_dir)
        print(f"Successfully loaded dataset wrapper. Found {len(full_dataset_raw._found_processed_paths)} potential files initially.")
    except RuntimeError as e:
        print(f"\nError during dataset initialization: {e}")
        print(f"Please ensure the directory '{args.processed_data_dir}/processed' exists and contains '.pt' files.")
        print("Run preprocess.py script correctly to generate these files.")
        exit(1)
    except FileNotFoundError:
         print(f"\nError: The specified processed data directory '{args.processed_data_dir}' or its 'processed' subdirectory was not found.")
         print("Please ensure the path is correct and preprocess.py has been run.")
         exit(1)

    if len(full_dataset_raw._found_processed_paths) == 0:
        print(f"Error: No '.pt' files found in the directory: {full_dataset_raw.processed_dir}")
        print("Please ensure preprocess.py ran successfully and generated output in the correct location.")
        exit(1)

    # --- Filtering Step ---
    MAX_NODES_THRESHOLD = args.max_nodes if args.max_nodes > 0 else float('inf')
    print(f"Filtering dataset: Keeping graphs with <= {args.max_nodes if args.max_nodes > 0 else 'infinity'} nodes...")

    filtered_indices = []
    valid_data_count = 0
    skipped_large = 0
    skipped_other = 0
    original_indices = list(range(len(full_dataset_raw))) # Use length based on found paths

    for i in original_indices:
        try:
            # Use get() which handles loading and basic checks (like NaN y)
            data = full_dataset_raw.get(i)

            if data is None:
                # get() already prints warnings for NaN or missing attrs
                skipped_other += 1
                continue

            # Check node count
            if hasattr(data, 'num_nodes'):
                if data.num_nodes <= MAX_NODES_THRESHOLD:
                    filtered_indices.append(i) # Keep index from the original dataset
                    valid_data_count += 1
                else:
                    skipped_large += 1
            else:
                # Should not happen if get() is robust, but good to check
                print(f"Warning: Skipping graph index {i} (PDB: {data.pdb_id if hasattr(data, 'pdb_id') else 'N/A'}) because it lacks 'num_nodes' attribute.")
                skipped_other += 1

        except (RuntimeError, FileNotFoundError, IndexError) as e:
            # Catch potential errors during get() that weren't handled internally
            print(f"Error processing index {i} during filtering: {e}. Skipping.")
            skipped_other += 1
        except Exception as e:
            print(f"Unexpected error processing index {i} during filtering: {e}. Skipping.")
            skipped_other += 1


    if not filtered_indices:
        raise RuntimeError(f"Filtering removed all graphs or no valid graphs found! Check threshold ({args.max_nodes}), data integrity, and paths.")

    # Create the filtered dataset using index_select (preferred PyG way for subsets)
    # index_select works directly on the Dataset object
    full_dataset = full_dataset_raw.index_select(filtered_indices)

    print(f"Filtered dataset size: {len(full_dataset)}")
    print(f"  Kept {valid_data_count} valid graphs within the size limit.")
    print(f"  Skipped {skipped_large} graphs due to exceeding node limit.")
    print(f"  Skipped {skipped_other} graphs due to loading errors, NaN targets, or missing attributes.")
    # --- End Filtering Step ---


    # Calculate input size based on preprocessing using the *filtered* dataset
    if len(full_dataset) == 0:
         raise ValueError("Dataset is empty after filtering.")
    # Get feature sizes from the first valid sample
    try:
        first_data = full_dataset.get(0)
        if first_data is None: # Should not happen if filtering worked, but safety check
             raise ValueError("Could not get a valid sample (index 0 returned None) from the filtered dataset.")
        in_size = first_data.x.shape[1] # Get actual feature size
        print(f"Determined input node feature size: {in_size}")
        # Check edge attributes more carefully
        use_edge_attr = not args.no_edge_attr and hasattr(first_data, 'edge_attr') and first_data.edge_attr is not None and first_data.edge_attr.numel() > 0
        num_edge_features = first_data.edge_attr.shape[1] if use_edge_attr else 0
        print(f"Using edge attributes: {use_edge_attr}, Edge feature size: {num_edge_features}")
    except Exception as e:
        print(f"Error getting feature sizes from the first sample of the filtered dataset: {e}")
        # Provide default or raise error depending on requirements
        # Example: Assuming default based on preprocess.py (less safe)
        # in_size = calculate_in_size() # Fallback, requires function definition
        # num_edge_features = 3 if not args.no_edge_attr else 0
        # print(f"Warning: Using fallback feature sizes. Node: {in_size}, Edge: {num_edge_features}")
        raise ValueError("Failed to determine feature sizes from filtered dataset.") from e


    # --- Train/Val/Test Split ---
    # Use the filtered dataset length and indices
    dataset_size = len(full_dataset)
    indices = list(range(dataset_size)) # Indices are now 0 to len(full_dataset)-1

    # Ensure splits are valid fractions
    if not (0.0 <= args.test_split < 1.0):
        raise ValueError(f"test_split must be between 0.0 and 1.0 (exclusive of 1), got {args.test_split}")
    if not (0.0 <= args.val_split < 1.0):
         raise ValueError(f"val_split must be between 0.0 and 1.0 (exclusive of 1), got {args.val_split}")

    num_test = int(dataset_size * args.test_split)
    # Val split is fraction of *remaining* data after test split
    num_remaining = dataset_size - num_test
    num_val = int(num_remaining * args.val_split)
    num_train = num_remaining - num_val

    # Handle cases where splits might be zero due to small dataset size
    if num_test == 0 and args.test_split > 0 and dataset_size > 0:
        print(f"Warning: Test split resulted in 0 samples (Dataset size: {dataset_size}). Adjusting to 1 test sample.")
        num_test = 1
        num_remaining = dataset_size - num_test
        # Recalculate val/train based on new remaining
        num_val = int(num_remaining * args.val_split)
        num_train = num_remaining - num_val
    if num_val == 0 and args.val_split > 0 and num_remaining > 0:
         print(f"Warning: Validation split resulted in 0 samples (Remaining size: {num_remaining}). Adjusting to 1 validation sample if possible.")
         num_val = 1 if num_remaining > 1 else 0 # Ensure train gets at least 1 if val=1
         num_train = num_remaining - num_val
    if num_train <= 0:
         raise ValueError(f"Train split resulted in {num_train} samples. Dataset might be too small for the chosen splits.")

    print(f"Data split: Train={num_train}, Validation={num_val}, Test={num_test}")

    # Perform shuffling *before* splitting
    random.shuffle(indices)

    # Split indices
    test_indices = indices[:num_test]
    val_indices = indices[num_test:num_test + num_val]
    train_indices = indices[num_test + num_val:]

    # Use index_select again for the final splits
    train_dataset = full_dataset.index_select(train_indices)
    # Create Subset objects only if indices exist
    val_dataset = full_dataset.index_select(val_indices) if val_indices else None
    test_dataset = full_dataset.index_select(test_indices) if test_indices else None


    print(f"Final Train samples: {len(train_dataset)}")
    print(f"Final Validation samples: {len(val_dataset) if val_dataset else 0}")
    print(f"Final Test samples: {len(test_dataset) if test_dataset else 0}")

    # Handle cases where splits might be empty after selection (shouldn't happen with checks above, but safer)
    if len(train_dataset) == 0: raise ValueError("Training set is empty after splitting!")
    if not val_dataset: print("Note: Validation set is empty.")
    if not test_dataset: print("Note: Test set is empty.")


    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_filter_none, drop_last=False) # drop_last can be False
    # Only create val/test loaders if the datasets are not empty
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_filter_none) if val_dataset else None
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_filter_none) if test_dataset else None


    # --- Model Initialization ---
    print("Initializing model...")
    # Pass relevant args from command line if needed
    model_kwargs = {
        'edge_dim': 32 if use_edge_attr else None, # Example: Tune edge_dim
        'k_hop': args.k_hop,
        # Add any other specific kwargs your layers might need
    }
    model = GraphTransformerRegression(
        in_size=in_size,
        out_dim=1, # Regression output
        d_model=args.d_model,
        num_heads=args.num_heads,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        num_layers=args.num_layers,
        batch_norm=args.batch_norm,
        gnn_type=args.gnn_type,
        se=args.se,
        use_edge_attr=use_edge_attr, # Use dynamically determined value
        num_edge_features=num_edge_features, # Use dynamically determined value
        global_pool=args.global_pool,
        **model_kwargs # Pass collected kwargs
    ).to(device)

    print(f"Model Parameters: {count_parameters(model):,}")

    # --- Loss and Optimizer ---
    criterion = nn.MSELoss() # Mean Squared Error for regression
    # criterion = nn.L1Loss() # Mean Absolute Error (alternative)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # Optional: Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

    # --- Training Loop ---
    print("\nStarting training...")
    best_val_rmse = float('inf')
    best_epoch = -1
    log_file = os.path.join(args.output_dir, "training_log.txt")

    with open(log_file, 'w') as f_log: # Open log file
        f_log.write("Epoch,Train_Loss,Val_Loss,Val_RMSE,Val_MAE,Epoch_Time_s,LR\n") # Write header

        for epoch in range(1, args.epochs + 1):
            epoch_start_time = time.time()
            train_loss, train_time = train_epoch(model, train_loader, criterion, optimizer, device)

            log_msg = f"Epoch {epoch}/{args.epochs} | Train Loss: {train_loss:.4f}"

            val_loss, val_rmse, val_mae = np.nan, np.nan, np.nan # Initialize validation metrics
            if val_loader:
                val_loss, val_rmse, val_mae = evaluate(model, val_loader, criterion, device)
                log_msg += f" | Val Loss: {val_loss:.4f} | Val RMSE: {val_rmse:.4f} | Val MAE: {val_mae:.4f}"

                # Update learning rate based on validation RMSE (or loss)
                if not np.isnan(val_rmse):
                     scheduler.step(val_rmse)
                elif not np.isnan(val_loss):
                     scheduler.step(val_loss) # Fallback to loss if RMSE is NaN

                # Save best model based on validation RMSE
                if not np.isnan(val_rmse) and val_rmse < best_val_rmse:
                    best_val_rmse = val_rmse
                    best_epoch = epoch
                    model_path = os.path.join(args.output_dir, "best_model.pt")
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': train_loss, # Save train loss too
                        'val_rmse': val_rmse,
                        'args': args # Save args for reproducibility
                    }, model_path)
                    log_msg += " | Best model saved!"
            else:
                 log_msg += " | (No validation set)"
                 # Save latest model if no validation
                 model_path = os.path.join(args.output_dir, "latest_model.pt")
                 torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': train_loss,
                        'args': args
                    }, model_path)
                 log_msg += " | Latest model saved!"

            current_lr = optimizer.param_groups[0]['lr']
            epoch_duration = time.time() - epoch_start_time
            log_msg += f" | LR: {current_lr:.2e} | Time: {epoch_duration:.2f}s"
            print(log_msg)
            # Write log entry to file
            f_log.write(f"{epoch},{train_loss:.6f},{val_loss:.6f},{val_rmse:.6f},{val_mae:.6f},{epoch_duration:.2f},{current_lr:.6e}\n")
            f_log.flush() # Ensure it's written immediately


    print("\nTraining finished.")
    if best_epoch != -1:
        print(f"Best Validation RMSE: {best_val_rmse:.4f} at Epoch {best_epoch}")
    else:
        print("No best model saved (no validation set or validation metrics were NaN).")

    # --- Final Evaluation on Test Set ---
    if test_loader:
        print("\nEvaluating on Test Set using best model...")
        # Load best model if it exists
        best_model_path = os.path.join(args.output_dir, "best_model.pt")
        if os.path.exists(best_model_path) and best_epoch != -1: # Check if best model was saved
            try:
                checkpoint = torch.load(best_model_path, map_location=device)
                # Ensure model architecture matches before loading state dict
                # (This check is implicitly done by PyTorch if keys mismatch)
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"Loaded best model from epoch {checkpoint.get('epoch', 'N/A')}")
            except Exception as e:
                print(f"Error loading best model state_dict: {e}. Evaluating with the final model state.")
        else:
            print("Warning: Best model checkpoint not found or validation was skipped/failed. Evaluating with the final model state.")

        test_loss, test_rmse, test_mae = evaluate(model, test_loader, criterion, device)
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test RMSE: {test_rmse:.4f}")
        print(f"Test MAE: {test_mae:.4f}")

        # Write test results to log file
        with open(log_file, 'a') as f_log:
             f_log.write("\nTest Set Evaluation\n")
             f_log.write(f"Test Loss,{test_loss:.6f}\n")
             f_log.write(f"Test RMSE,{test_rmse:.6f}\n")
             f_log.write(f"Test MAE,{test_mae:.6f}\n")

    else:
        print("\nNo test set provided for final evaluation.")

    print(f"\nResults and checkpoints saved in: {args.output_dir}")
    print(f"Training log saved to: {log_file}")