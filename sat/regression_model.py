import torch
from torch import nn
import torch_geometric.nn as gnn
# Make sure to import from your actual layers file location
# If layers.py is in the same directory:
from layers import TransformerEncoderLayer
# Or if it's in a subpackage 'my_package':
# from my_package.layers import TransformerEncoderLayer
from einops import repeat

class GraphTransformerEncoder(nn.TransformerEncoder):
    # --- (Keep this class exactly as in your models.py) ---
    def forward(self, x, edge_index, complete_edge_index,
            subgraph_node_index=None, subgraph_edge_index=None,
            subgraph_edge_attr=None, subgraph_indicator_index=None, edge_attr=None, degree=None,
            ptr=None, return_attn=False):
        output = x

        for mod in self.layers:
            output = mod(output, edge_index, complete_edge_index,
                edge_attr=edge_attr, degree=degree,
                subgraph_node_index=subgraph_node_index,
                subgraph_edge_index=subgraph_edge_index,
                subgraph_indicator_index=subgraph_indicator_index,
                subgraph_edge_attr=subgraph_edge_attr,
                ptr=ptr,
                return_attn=return_attn
            )
        if self.norm is not None:
            output = self.norm(output)
        return output


class GraphTransformerRegression(nn.Module):
    # Changed num_class to out_dim=1 for regression
    def __init__(self, in_size, out_dim=1, d_model=128, num_heads=8,
                 dim_feedforward=512, dropout=0.1, num_layers=4,
                 batch_norm=True, abs_pe=False, abs_pe_dim=0,
                 gnn_type="gcn", se="gnn", use_edge_attr=True, num_edge_features=3, # Adjusted defaults
                 in_embed=False, edge_embed=False, use_global_pool=True, max_seq_len=None, # Removed max_seq_len logic
                 global_pool='mean', **kwargs):
        super().__init__()

        self.abs_pe = abs_pe
        self.abs_pe_dim = abs_pe_dim
        if abs_pe and abs_pe_dim > 0:
            self.embedding_abs_pe = nn.Linear(abs_pe_dim, d_model)

        # Node embedding
        if in_embed:
            if isinstance(in_size, int):
                # Usually for discrete features like atom types if used as indices
                self.embedding = nn.Embedding(in_size, d_model)
            elif isinstance(in_size, nn.Module):
                self.embedding = in_size
            else:
                raise ValueError("Not implemented!")
        else:
            # Assumes continuous input features from get_atom_features
            self.embedding = nn.Linear(in_features=in_size,
                                       out_features=d_model,
                                       bias=False)

        self.use_edge_attr = use_edge_attr
        if use_edge_attr:
            edge_dim = kwargs.get('edge_dim', 32) # Can be tuned
            if edge_embed:
                # If edge features were discrete categories
                if isinstance(num_edge_features, int):
                    self.embedding_edge = nn.Embedding(num_edge_features, edge_dim)
                else:
                    raise ValueError("Not implemented!")
            else:
                 # Assumes continuous edge features (like the one-hot encoding)
                 # or distances if you were to use them
                self.embedding_edge = nn.Linear(in_features=num_edge_features,
                    out_features=edge_dim, bias=False)
            kwargs['edge_dim'] = edge_dim # Pass edge_dim to the attention layers
        else:
            kwargs['edge_dim'] = None

        self.gnn_type = gnn_type
        self.se = se
        encoder_layer = TransformerEncoderLayer(
            d_model, num_heads, dim_feedforward, dropout, batch_norm=batch_norm,
            gnn_type=gnn_type, se=se, use_edge_attr=use_edge_attr, **kwargs) # Pass use_edge_attr here if needed by layer
        self.encoder = GraphTransformerEncoder(encoder_layer, num_layers)

        self.global_pool = global_pool
        self.use_global_pool = use_global_pool
        if use_global_pool:
            if global_pool == 'mean':
                self.pooling = gnn.global_mean_pool
            elif global_pool == 'add':
                self.pooling = gnn.global_add_pool
            elif global_pool == 'cls':
                self.cls_token = nn.Parameter(torch.randn(1, d_model))
                self.pooling = None # CLS token is appended and then selected
            else:
                 raise ValueError(f"Unsupported global pooling: {global_pool}")
        else:
            # Regression usually requires a graph-level output
            print("Warning: use_global_pool is False. Regression typically needs a graph-level embedding.")
            self.pooling = None


        # --- Regression Head ---
        # Simplified classifier for regression: maps pooled features to a single output value
        # Removed ReLU before the final layer, common practice for regression
        self.regressor_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2), # Optional intermediate layer
            nn.ReLU(),
            nn.Dropout(dropout), # Optional dropout
            nn.Linear(d_model // 2, out_dim) # Output layer with out_dim (usually 1)
        )
        # Alternative simpler head:
        # self.regressor_head = nn.Linear(d_model, out_dim)


    def forward(self, data, return_attn=False):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        # Make sure batch attribute exists

        node_depth = data.node_depth if hasattr(data, "node_depth") else None # Assuming this isn't used based on preprocess.py

        # --- Feature Handling (same as original, ensure compatibility) ---
        if self.se == "khopgnn":
             # Ensure these attributes are loaded correctly by the DataLoader
            subgraph_node_index = data.subgraph_node_idx if hasattr(data, "subgraph_node_idx") else None
            subgraph_edge_index = data.subgraph_edge_index if hasattr(data, "subgraph_edge_index") else None
            subgraph_indicator_index = data.subgraph_indicator if hasattr(data, "subgraph_indicator") else None
            subgraph_edge_attr = data.subgraph_edge_attr if hasattr(data, "subgraph_edge_attr") else None
        else:
            subgraph_node_index = None
            subgraph_edge_index = None
            subgraph_indicator_index = None
            subgraph_edge_attr = None

        # Optional features (handle if not present in data)
        complete_edge_index = data.complete_edge_index if hasattr(data, 'complete_edge_index') else None
        abs_pe = data.abs_pe if hasattr(data, 'abs_pe') else None
        degree = data.degree if hasattr(data, 'degree') else None

        # --- Embeddings ---
        # Node embedding
        h = self.embedding(x) if node_depth is None else self.embedding(x, node_depth.view(-1,))

        # Optional absolute PE embedding
        if self.abs_pe and abs_pe is not None:
            abs_pe_emb = self.embedding_abs_pe(abs_pe)
            h = h + abs_pe_emb

        # Edge embedding
        if self.use_edge_attr and edge_attr is not None:
            # Ensure edge_attr has the correct feature dimension (e.g., 3)
            edge_attr_emb = self.embedding_edge(edge_attr)
            if subgraph_edge_attr is not None:
                 # Ensure subgraph_edge_attr also has the correct dim
                subgraph_edge_attr = self.embedding_edge(subgraph_edge_attr)
        else:
            edge_attr_emb = None
            subgraph_edge_attr = None # Ensure it's None if not used


        # --- CLS Token Logic (if used) ---
        bsz = data.num_graphs # Use num_graphs for clarity
        if self.global_pool == 'cls' and self.use_global_pool:
            cls_tokens = repeat(self.cls_token, '() d -> b d', b=bsz)
             # We need to adjust batch indices and edge indices if adding CLS tokens
             # This requires careful handling of indices, often simpler to use mean/add pooling first

             # Simplified: Add CLS tokens to node features 'h'
            h = torch.cat([h, cls_tokens], dim=0)
            # Add corresponding entries to the batch vector for the CLS tokens
            cls_batch = torch.arange(bsz, device=batch.device)
            batch = torch.cat([batch, cls_batch], dim=0)

            # *** Important: complete_edge_index and k-hop structures need careful updates
            # if CLS tokens are added *before* the encoder. It might be easier to add
            # CLS *after* the encoder if you face issues here.
            # For now, assume the encoder handles the augmented batch correctly.
            # (The original implementation seemed to handle this, let's proceed)
            # Update: Need ptr for padding if using cls token before encoder. Reverting to mean/add pooling is safer start.
            if ptr is None and self.encoder.needs_ptr_for_cls_padding: # Hypothetical check
                 raise ValueError("CLS token prepending requires `ptr` for padding logic in encoder.")


        # --- Graph Transformer Encoder ---
        output = self.encoder(
            h,                  # Use 'h' which includes embeddings
            edge_index,
            complete_edge_index,
            edge_attr=edge_attr_emb, # Use embedded edge attributes
            degree=degree,
            subgraph_node_index=subgraph_node_index,
            subgraph_edge_index=subgraph_edge_index,
            subgraph_indicator_index=subgraph_indicator_index,
            subgraph_edge_attr=subgraph_edge_attr, # Use embedded subgraph edge attributes
            ptr=data.ptr if hasattr(data, 'ptr') else None, # Pass ptr if available/needed
            return_attn=return_attn
        )

        # --- Readout ---
        if self.use_global_pool:
            if self.global_pool == 'cls':
                # Select the features corresponding to the CLS tokens
                # Assumes CLS tokens were added at the end of the batch dimension
                output = output[data.num_nodes:] # Select the last 'bsz' entries
            else:
                # Apply mean or add pooling based on the original batch assignment
                output = self.pooling(output, batch) # Pass the correct batch vector

        # --- Regression Head ---
        # Final prediction
        prediction = self.regressor_head(output)

        # Ensure output is squeezed if out_dim is 1, matching target shape [batch_size]
        if prediction.shape[-1] == 1:
            return prediction.squeeze(-1)
        else:
            return prediction

# --- END OF FILE models_regression.py ---


