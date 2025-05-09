# Model Architectures

This document provides a high-level overview of the three models used for binding affinity prediction.

## 1. GCNRegression (`gcn_regression_model.py`)

**Type**: Graph Convolutional Network (GCN) or a GINE-like model if `use_edge_features=True`.

**Input**:
- Node features: `data.x` (atom features from `dataset_gcn.py`).
- Edge index: `data.edge_index` (combined intra-ligand, intra-pocket, and inter-molecular connections).
- Edge attributes: `data.edge_attr` (optional, e.g., one-hot encoding for [intra-bond, inter-interaction] type).
- Batch assignment: `data.batch`.

**Core Architecture**:
1.  **Graph Convolutional Layers (Multiple)**:
    *   If `use_edge_features=False`: Standard `GCNConv` layers are used. Each layer aggregates features from neighboring nodes and applies a linear transformation, followed by activation and batch normalization.
        `H_out = ReLU(BatchNorm(GCNConv(H_in, edge_index)))`
    *   If `use_edge_features=True`: `GINEConv` layers are used (as per the GCNRegression implementation, GCNConv is specified but GINEConv is used if edge_dim is given to the custom GCNConv in gnn_layers.py or if GINEConv is used directly in gcn_regression_model.py. The provided `gcn_regression_model.py` uses `GINEConv` from `torch_geometric.nn` if `use_edge_features` and `edge_dim` are provided, which internally takes an MLP for node feature transformation.). GINEConv incorporates edge features into the message passing scheme, typically by transforming edge attributes and combining them with node features.
        `H_out = ReLU(BatchNorm(GINEConv(MLP(H_in), edge_index, edge_attr=edge_attr)))`
    *   Dropout is applied between layers.

2.  **Global Pooling**:
    *   After the convolutional stack, node embeddings are aggregated into a single graph-level embedding using `global_mean_pool` or `global_add_pool`.
    *   Input: Node embeddings `H_final` from the last conv layer, `data.batch`.
    *   Output: Graph embeddings `h_graph` (batch_size, hidden_dim).

3.  **Regression Head (Feed-Forward Network)**:
    *   A multi-layer perceptron (MLP) maps the graph embedding to the final binding affinity prediction.
    *   Structure: `Linear -> ReLU -> Dropout -> Linear -> Output (scalar)`.

**Simplified Diagram**:
Use code with caution.
Markdown
Input Graph (Nodes, Edges, Edge Attrs)
|
V
[ GCN/GINE Conv Layer + BN + ReLU (+ Dropout) ] x NumLayers
|
V
Global Pooling (Mean or Add) --> Graph-level Embedding
|
V
Regression MLP (Linear layers + Activations + Dropout)
|
V
Predicted Affinity (Scalar)
## 2. GINRegression (`gin_regression_model.py`)

**Type**: Graph Isomorphism Network (GIN) or Graph Isomorphism Network with Edge Features (GINE).

**Input**: Same as `GCNRegression`.

**Core Architecture**:
1.  **Graph Isomorphism Layers (Multiple)**:
    *   Each layer uses either `GINConv` (if `use_edge_features=False`) or `GINEConv` (if `use_edge_features=True`).
    *   `GINConv`: Aggregates messages using an MLP applied to the sum of the central node's features (multiplied by `1+epsilon`) and its neighbors' features. `epsilon` can be learnable.
        `H_out = ReLU(BatchNorm(GINConv(MLP_GIN(H_in), edge_index)))`
    *   `GINEConv`: Similar to GIN but incorporates edge features into the message aggregation, typically by transforming edge attributes and adding them to messages passed from neighbors.
        `H_out = ReLU(BatchNorm(GINEConv(MLP_GINE(H_in), edge_index, edge_attr=edge_attr)))`
    *   The MLP within GIN/GINE is typically a 2-layer network (`Linear -> ReLU -> Linear`).
    *   Dropout is applied between layers.

2.  **Global Pooling**:
    *   Same as `GCNRegression` (e.g., `global_mean_pool` or `global_add_pool`).

3.  **Regression Head (Feed-Forward Network)**:
    *   Same as `GCNRegression`.

**Simplified Diagram**:
Use code with caution.
Input Graph (Nodes, Edges, Edge Attrs)
|
V
[ GIN/GINE Conv Layer (with internal MLP) + BN + ReLU (+ Dropout) ] x NumLayers
|
V
Global Pooling (Mean or Add) --> Graph-level Embedding
|
V
Regression MLP
|
V
Predicted Affinity (Scalar)
## 3. GraphTransformerRegression (`regression_model.py`)

**Type**: Graph Transformer with Structure-Aware Self-Attention.

**Input (varies based on `se` structure extractor type)**:
- Node features: `data.x` (atom features from `dataset_GIGN.py`).
- `data.edge_index`, `data.edge_attr` (for `se='gnn'`).
- `data.complete_edge_index` (for full graph attention).
- `data.batch`, `data.ptr`.
- `data.abs_pe` (optional absolute positional encodings).
- If `se='khopgnn'`: `data.subgraph_node_idx`, `data.subgraph_edge_index`, `data.subgraph_indicator`, `data.subgraph_edge_attr`. (These come from `data.py:GraphDataset` if used, but `train_regression.py` uses `dataset_GIGN.py` which doesn't prepare these specific subgraph attributes. If `khopgnn` SE is selected with `dataset_GIGN.py`, it will fail unless `dataset_GIGN.py` is modified or a compatible dataset is used).
- If `se='gign'`: `data.edge_index_intra`, `data.edge_index_inter`, `data.pos` (node coordinates).

**Core Architecture**:
1.  **Initial Node Embedding**:
    *   Raw node features `data.x` are embedded to `d_model` dimensions (e.g., via `nn.Linear` or `nn.Embedding`).
    *   Optional: Add learned absolute positional encodings.
2.  **Optional Edge Embedding**:
    *   If `use_edge_attr=True`, raw edge attributes (`data.edge_attr`, `data.subgraph_edge_attr`) are embedded.
3.  **Graph Transformer Encoder Layers (Multiple)**: (Implemented in `regression_model.py:GraphTransformerEncoder` using `layers.py:TransformerEncoderLayer`)
    *   Each layer contains:
        *   **a. Structure-Aware Multi-Head Self-Attention (`layers.Attention`)**:
            *   **Structure Extraction (SE)**: Node embeddings `H` are processed by an SE to get `H_struct`.
                *   `se='gnn'`: A GNN (e.g., GCN, GIN defined by `gnn_type`) processes `H` using the graph's `edge_index` and `edge_attr`.
                *   `se='khopgnn'`: (As noted above, requires specific data attributes not in `dataset_GIGN.py` by default). A GNN processes k-hop subgraphs.
                *   `se='gign'`: A stack of HIL (Heterogeneous Interaction Layers from `HIL.py`) processes `H` using `edge_index_intra`, `edge_index_inter`, and `pos` to model 3D interactions.
            *   **Attention Mechanism**: Q and K are derived from `H_struct`, V from `H`. Scaled dot-product attention is computed, potentially over `complete_edge_index` (sparse) or densely.
        *   **b. Add & Norm**: Residual connection and LayerNorm/BatchNorm.
        *   **c. Feed-Forward Network (FFN)**: Position-wise MLP.
        *   **d. Add & Norm**.
4.  **Global Pooling**:
    *   If `use_global_pool=True`: `global_mean_pool`, `global_add_pool`, or a [CLS] token strategy is used to get a graph-level embedding from the final node embeddings.
5.  **Regression Head**:
    *   MLP maps the graph-level embedding to the affinity prediction.

**Simplified Diagram (focus on `se='gign'`)**:
Use code with caution.
Input (Nodes, Coords, Intra/Inter Edges, etc.)
|
V
Initial Node Embedding (+ AbsPE) --> H (N, d_model)
|
|---- Repeated ----|
| H_prev |
| | | Structure Extractor ('gign'): H_prev + HIL layers --> H_struct
| V | Q, K from H_struct; V from H_prev
| Multi-Head Attention (H_struct, H_prev) --> Attn_Output
| | |
| Add & Norm |
| | |
| Feed-Forward Net |
| | |
| Add & Norm |
| | |
| H_next |
|------------------| x NumLayers
|
V
Final Node Embeddings (N, d_model)
|
V
Global Pooling (Mean, Add, CLS) --> Graph-level Embedding (B, d_model)
|
V
Regression MLP --> Predicted Affinity (Scalar)