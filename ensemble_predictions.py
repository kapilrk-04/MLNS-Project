import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
from math import sqrt
import os

def calculate_metrics(y_true, y_pred, model_name="Model"):
    if len(y_true) == 0 or len(y_pred) == 0:
        print(f"Warning: Empty true or predicted values for {model_name}.")
        return {"rmse": np.nan, "mae": np.nan, "pearsonr": np.nan}
    if len(y_true) != len(y_pred):
        print(f"Warning: Mismatch in length of true ({len(y_true)}) and predicted ({len(y_pred)}) values for {model_name}.")
        return {"rmse": np.nan, "mae": np.nan, "pearsonr": np.nan}

    rmse = sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    
    pr = np.nan
    if len(y_true) >= 2: # Pearson R requires at least 2 samples
        try:
            pr_val, _ = pearsonr(y_true, y_pred)
            if not np.isnan(pr_val):
                pr = pr_val
        except ValueError:
            pass # pr remains np.nan
            
    return {"rmse": rmse, "mae": mae, "pearsonr": pr}

def main_ensemble():
    # Define paths to the prediction files. Adjust these if your checkpoint/output dirs differ.
    gcn_preds_path = './checkpoints_gcn/gcn_regression_test_predictions.csv'
    gin_preds_path = './checkpoints_gin/gin_regression_test_predictions.csv'
    # The GIGN-Transformer path depends on args.output_dir used in train_regression.py
    gign_transformer_preds_path = './output_gign_transformer_example/gign_transformer_test_predictions.csv'

    pred_files = {
        "gcn": gcn_preds_path,
        "gin": gin_preds_path,
        "gign_transformer": gign_transformer_preds_path
    }

    loaded_dfs = {}
    for model_name, path in pred_files.items():
        if not os.path.exists(path):
            print(f"Error: Prediction file for {model_name} not found at {path}")
            print("Please ensure all three models have been trained and their test predictions saved.")
            return
        loaded_dfs[model_name] = pd.read_csv(path)
        print(f"Loaded predictions for {model_name} from {path} ({len(loaded_dfs[model_name])} rows)")

    # Merge the dataframes on 'pdbid'
    # Start with GCN, then merge GIN, then GIGN-Transformer
    df_merged = loaded_dfs['gcn'].rename(columns={'y_pred': 'y_pred_gcn'})
    
    df_merged = pd.merge(
        df_merged,
        loaded_dfs['gin'][['pdbid', 'y_pred']].rename(columns={'y_pred': 'y_pred_gin'}),
        on='pdbid',
        how='inner' # Use 'inner' to keep only common PDBs
    )
    df_merged = pd.merge(
        df_merged,
        loaded_dfs['gign_transformer'][['pdbid', 'y_pred']].rename(columns={'y_pred': 'y_pred_gign_transformer'}),
        on='pdbid',
        how='inner'
    )
    
    # Check if y_true is consistent (it should be if test sets are identical)
    # If multiple y_true columns exist due to merge, select one or verify consistency.
    # Assuming 'y_true' from the first loaded df (GCN's) is the reference.
    
    print(f"Successfully merged predictions for {len(df_merged)} common PDB IDs.")
    if df_merged.empty:
        print("No common PDB IDs found across the prediction files. Exiting.")
        return

    y_true_values = df_merged['y_true'].values
    
    print("\n--- Individual Model Performance (on common test subset) ---")
    metrics_gcn = calculate_metrics(y_true_values, df_merged['y_pred_gcn'].values, "GCN")
    print(f"GCN:          RMSE={metrics_gcn['rmse']:.4f}, MAE={metrics_gcn['mae']:.4f}, R={metrics_gcn['pearsonr']:.4f}")
    
    metrics_gin = calculate_metrics(y_true_values, df_merged['y_pred_gin'].values, "GIN")
    print(f"GIN:          RMSE={metrics_gin['rmse']:.4f}, MAE={metrics_gin['mae']:.4f}, R={metrics_gin['pearsonr']:.4f}")

    metrics_gign_transformer = calculate_metrics(y_true_values, df_merged['y_pred_gign_transformer'].values, "GIGN-Transformer")
    print(f"GIGN-Trans:   RMSE={metrics_gign_transformer['rmse']:.4f}, MAE={metrics_gign_transformer['mae']:.4f}, R={metrics_gign_transformer['pearsonr']:.4f}")

    # --- Simple Averaging Ensemble ---
    df_merged['y_pred_ensemble_avg'] = df_merged[['y_pred_gcn', 'y_pred_gin', 'y_pred_gign_transformer']].mean(axis=1)
    
    print("\n--- Ensemble Model Performance (Simple Average) ---")
    metrics_ensemble_avg = calculate_metrics(y_true_values, df_merged['y_pred_ensemble_avg'].values, "Average Ensemble")
    print(f"Avg Ensemble: RMSE={metrics_ensemble_avg['rmse']:.4f}, MAE={metrics_ensemble_avg['mae']:.4f}, R={metrics_ensemble_avg['pearsonr']:.4f}")

    # --- Weighted Averaging Ensemble (Example) ---
    # Weights should ideally be tuned on a separate validation set.
    # Here's an example of how to apply them if you had them.
    # For instance, if GIGN-Transformer performed best on validation, GIN second, GCN third:
    # weights = np.array([0.2, 0.3, 0.5]) # Example: GCN, GIN, GIGN-T
    
    # A simple heuristic for weights could be inverse of validation RMSE (normalized)
    # This is just illustrative. Proper hyperparameter optimization for weights is better.
    val_rmses = np.array([
        metrics_gcn.get('rmse', np.inf), 
        metrics_gin.get('rmse', np.inf), 
        metrics_gign_transformer.get('rmse', np.inf)
    ])
    
    # Avoid division by zero or issues with np.inf
    valid_rmses = np.isfinite(val_rmses)
    if np.any(valid_rmses):
        # Calculate inverse RMSE, handle non-finite values by giving them zero weight contribution initially
        inv_rmses = np.zeros_like(val_rmses)
        inv_rmses[valid_rmses] = 1.0 / val_rmses[valid_rmses]
        
        if np.sum(inv_rmses) > 0: # Normalize to sum to 1
            weights = inv_rmses / np.sum(inv_rmses)
            print(f"\nCalculated heuristic weights based on test RMSEs (for illustration):")
            print(f"  GCN: {weights[0]:.3f}, GIN: {weights[1]:.3f}, GIGN-T: {weights[2]:.3f}")

            df_merged['y_pred_ensemble_weighted'] = np.average(
                df_merged[['y_pred_gcn', 'y_pred_gin', 'y_pred_gign_transformer']].values,
                axis=1,
                weights=weights
            )
            metrics_ensemble_weighted = calculate_metrics(y_true_values, df_merged['y_pred_ensemble_weighted'].values, "Weighted Ensemble")
            print(f"\nWeighted Ensemble (Test RMSE-based heuristic): RMSE={metrics_ensemble_weighted['rmse']:.4f}, MAE={metrics_ensemble_weighted['mae']:.4f}, R={metrics_ensemble_weighted['pearsonr']:.4f}")
        else:
            print("\nCould not calculate heuristic weights (e.g., all RMSEs were non-finite). Skipping weighted ensemble.")
    else:
        print("\nAll model RMSEs were non-finite. Skipping weighted ensemble.")

if __name__ == '__main__':
    main_ensemble()