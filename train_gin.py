# --- START OF FILE train_gin_regression.py ---
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
import argparse
import os
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from math import sqrt
from scipy.stats import pearsonr as scipy_pearsonr

# Import model and dataset
from gin_regression_model import GINRegression
from dataset_gcn import GCNGraphDataset # Reusing the GCN dataset script as it's compatible

# --- Evaluation Metrics (same as before) ---
def calculate_rmse(y_true, y_pred):
    return sqrt(nn.functional.mse_loss(y_true, y_pred).item())

def calculate_mae(y_true, y_pred):
    return nn.functional.l1_loss(y_true, y_pred).item()

def calculate_pearsonr(y_true, y_pred):
    y_true_np = y_true.cpu().numpy()
    y_pred_np = y_pred.cpu().numpy()
    if len(y_true_np) < 2 or len(y_pred_np) < 2: return 0.0
    try:
        r, _ = scipy_pearsonr(y_true_np, y_pred_np)
        return r if not np.isnan(r) else 0.0
    except ValueError:
        return 0.0

# --- Training and Evaluation Functions (same as before) ---
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    processed_graphs = 0
    for batch in loader:
        if batch is None: continue
        batch = batch.to(device)
        optimizer.zero_grad()
        predictions = model(batch)
        loss = criterion(predictions, batch.y.squeeze(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
        processed_graphs += batch.num_graphs
    return total_loss / processed_graphs if processed_graphs > 0 else 0.0

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds, all_targets = [], []
    processed_graphs = 0
    with torch.no_grad():
        for batch in loader:
            if batch is None: continue
            batch = batch.to(device)
            predictions = model(batch)
            target = batch.y.squeeze(-1)
            loss = criterion(predictions, target)
            total_loss += loss.item() * batch.num_graphs
            processed_graphs += batch.num_graphs
            all_preds.append(predictions.cpu())
            all_targets.append(target.cpu())
    if processed_graphs == 0:
        return {"loss": float('inf'), "rmse": float('inf'), "mae": float('inf'), "r": 0.0}

    avg_loss = total_loss / processed_graphs
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    return {
        "loss": avg_loss,
        "rmse": calculate_rmse(all_targets, all_preds),
        "mae": calculate_mae(all_targets, all_preds),
        "r": calculate_pearsonr(all_targets, all_preds)
    }

def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda': torch.cuda.manual_seed(args.seed)
    print(f"Using device: {device}")

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    print("Loading data index...")
    try:
        df = pd.read_csv(args.label_file)
    except FileNotFoundError:
        print(f"Error: Label file not found at {args.label_file}"); return
    if args.target_col not in df.columns:
        print(f"Error: Target column '{args.target_col}' not found. Available: {df.columns.tolist()}"); return

    train_df, test_val_df = train_test_split(df, test_size=0.2, random_state=args.seed)
    val_df, test_df = train_test_split(test_val_df, test_size=0.5, random_state=args.seed)
    print(f"Dataset splits: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    print("Initializing GIN datasets (using GCN dataset script)...")
    # Using GCNGraphDataset as the structure is compatible
    train_dataset = GCNGraphDataset(
        args.data_dir, train_df, dis_threshold=args.distance, graph_type=args.graph_type,
        num_process=args.num_workers, create=args.create_graphs,
        complex_id_col=args.complex_id_col, target_col=args.target_col
    )
    val_dataset = GCNGraphDataset(
        args.data_dir, val_df, dis_threshold=args.distance, graph_type=args.graph_type,
        create=False, complex_id_col=args.complex_id_col, target_col=args.target_col
    )
    test_dataset = GCNGraphDataset(
        args.data_dir, test_df, dis_threshold=args.distance, graph_type=args.graph_type,
        create=False, complex_id_col=args.complex_id_col, target_col=args.target_col
    )

    if len(train_dataset) == 0 or train_dataset.n_features is None or train_dataset.n_features <= 0:
        print("Error: Training dataset is empty or failed to determine features. Exiting."); return

    in_channels = train_dataset.n_features
    edge_dim = train_dataset.num_edge_features if args.use_edge_features else None
    print(f"Model input features: {in_channels}, Edge features: {edge_dim if edge_dim else 'Not Used (GIN)' if not args.use_edge_features else 'Used (GINE)'}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=GCNGraphDataset.collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=GCNGraphDataset.collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=GCNGraphDataset.collate_fn)

    model = GINRegression(
        in_channels=in_channels,
        hidden_channels=args.hidden_channels,
        out_channels=1,
        num_layers=args.num_layers,
        dropout=args.dropout,
        use_edge_features=args.use_edge_features,
        edge_dim=edge_dim, # Pass actual edge dimension if using edge features (for GINE)
        global_pool_type=args.global_pool,
        train_eps=args.train_eps
    ).to(device)
    print(f"GIN/GINE Model: \n{model}")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

    print("Starting GIN/GINE training...")
    best_val_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_metrics["loss"])
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch}/{args.epochs} | Time: {epoch_time:.2f}s | Train Loss: {train_loss:.4f} | Val Loss: {val_metrics['loss']:.4f} | Val RMSE: {val_metrics['rmse']:.4f} | Val R: {val_metrics['r']:.4f}")

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            save_path = os.path.join(args.checkpoint_dir, f"{args.model_name}_best.pth")
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'loss': best_val_loss}, save_path)
            print(f"*** Best GIN/GINE model saved at epoch {epoch} ***")

    print("\nTraining finished.")
    best_model_path = os.path.join(args.checkpoint_dir, f"{args.model_name}_best.pth")
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Best GIN/GINE model from epoch {checkpoint['epoch']} loaded for test evaluation.")
        test_metrics = evaluate(model, test_loader, criterion, device)
        print("\n--- GIN/GINE Test Set Performance (Best Model) ---")
        for metric, value in test_metrics.items(): print(f"{metric.upper()}: {value:.4f}")
    else:
        print("Warning: Best GIN/GINE model checkpoint not found.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GIN/GINE for Binding Affinity Regression')
    # Data args (same as GCN)
    parser.add_argument('--data_dir', type=str, default='data/PDBbind_v2020_other_PL/v2020-other-PL', help='Directory of complex subfolders')
    parser.add_argument('--label_file', type=str, default='data/binding_data.csv', help='CSV with PDB IDs and labels')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints_gin', help='Directory for checkpoints')
    parser.add_argument('--model_name', type=str, default='gin_regression', help='Name for model checkpoints')
    parser.add_argument('--graph_type', type=str, default='Graph_GCN', help='Prefix for saved graph files (can reuse GCN graphs)') # Reuse GCN graphs
    parser.add_argument('--distance', type=float, default=5.0, help='Distance threshold')
    parser.add_argument('--complex_id_col', type=str, default='pdbid', help='Complex ID column in CSV')
    parser.add_argument('--target_col', type=str, default='-logkd/ki', help='Target value column in CSV')
    parser.add_argument('--create_graphs', action='store_true', help='Generate PyG graph files if they dont exist')

    # Training args (same as GCN)
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda or cpu)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num_workers', type=int, default=0, help='DataLoader workers')

    # Model args (GIN specific)
    parser.add_argument('--hidden_channels', type=int, default=128, help='GIN/GINE hidden channels')
    parser.add_argument('--num_layers', type=int, default=3, help='Number of GIN/GINE layers')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
    parser.add_argument('--use_edge_features', action='store_true', default=True, help='Use edge features (GINEConv) vs no edge features (GINConv)')
    parser.add_argument('--train_eps', action='store_true', default=False, help='Make GIN/GINE epsilon parameter learnable')
    parser.add_argument('--global_pool', type=str, default='mean', choices=['mean', 'add'], help='Global pooling')

    args = parser.parse_args()
    main(args)
# --- END OF FILE train_gin_regression.py ---