import torch
import numpy as np
from torch import nn
from torch_scatter import scatter_add, scatter_mean, scatter_max
import torch_geometric.nn as gnn
import torch_geometric.utils as utils
from einops import rearrange
from utils import pad_batch, unpad_batch 
from gnn_layers import get_simple_gnn_layer, EDGE_GNN_TYPES
import torch.nn.functional as F

# Attempt to import HIL, assuming HIL.py is in the same directory or Python path
try:
    from HIL import HIL
except ImportError:
    # Fallback if HIL.py is not directly accessible, e.g. if running scripts from different dirs
    # This might require adjusting PYTHONPATH or file structure for robust imports
    print("Warning: Could not import HIL directly. Ensure HIL.py is accessible.")
    # As a placeholder, define a dummy HIL if real import fails, to allow syntax checking.
    # This should be replaced by ensuring correct import paths.
    if 'HIL' not in globals():
        class HIL(nn.Module):
            def __init__(self, in_channels, out_channels, **kwargs): super().__init__(); self.fc = nn.Linear(in_channels, out_channels)
            def forward(self, x, *args): return self.fc(x)


# Helper for padding, if utils.py is not available.
# These are simplified versions. Robust padding might require more careful handling of batch structures.
def pad_batch(tensor, ptr, return_mask=False):
    # Simplified: Assumes tensor is [N, D] and ptr indicates start of each graph.
    # This is a placeholder if the original `utils.pad_batch` is unavailable.
    # For robust CLS token handling, a proper padding mechanism is essential.
    # For mean/add pooling, this might not be strictly necessary for the main Transformer path.
    if ptr is None: # Cannot pad without ptr
        if return_mask:
            return tensor.unsqueeze(0), torch.ones(1, tensor.size(0), dtype=torch.bool, device=tensor.device) # Treat as one large graph
        return tensor.unsqueeze(0)

    num_graphs = len(ptr) - 1
    max_len = (ptr[1:] - ptr[:-1]).max()
    padded_tensor = torch.zeros(num_graphs, max_len, tensor.shape[-1], device=tensor.device)
    mask = torch.ones(num_graphs, max_len, dtype=torch.bool, device=tensor.device)
    for i in range(num_graphs):
        start, end = ptr[i], ptr[i+1]
        graph_len = end - start
        padded_tensor[i, :graph_len] = tensor[start:end]
        mask[i, graph_len:] = False # Mask out padding
    if return_mask:
        return padded_tensor, mask
    return padded_tensor

def unpad_batch(tensor, ptr):
    # Simplified unpadding, placeholder if original `utils.unpad_batch` is unavailable.
    if ptr is None: # Cannot unpad without ptr
        return tensor.squeeze(0)

    return torch.cat([tensor[i, :(ptr[i+1]-ptr[i])] for i in range(len(ptr)-1)], dim=0)


class GIGNStructureExtractor(nn.Module):
    def __init__(self, d_model, num_gign_layers, gign_dropout=0.1): # d_model is embed_dim from Transformer
        super().__init__()
        # Input x is already d_model dimensional from GraphTransformerRegression's initial embedding
        # GIGN's HIL layers will operate on this d_model dimension.
        # Original GIGN has a lin_node before HIL. Here, initial embedding in main model serves a similar purpose.
        # Or, we can add one here if desired for GIGN SE specifically.
        # Let's keep it simple: HIL layers operate on d_model directly.
        # If specific GIGN pre-projection is needed:
        # self.pre_gign_lin = nn.Sequential(nn.Linear(d_model, d_model), nn.SiLU())
        
        self.gconvs = nn.ModuleList()
        for _ in range(num_gign_layers):
            # HIL's in_channels and out_channels will be d_model
            self.gconvs.append(HIL(d_model, d_model)) # Dropout is within HIL if defined there

        # Output dimension is d_model

    def forward(self, x, edge_index_intra, edge_index_inter, pos):
        h = x # x is (N, d_model)
        # if hasattr(self, 'pre_gign_lin'): h = self.pre_gign_lin(h)

        for gconv in self.gconvs:
            h = gconv(h, edge_index_intra, edge_index_inter, pos)
        return h


class Attention(gnn.MessagePassing):
    """Multi-head Structure-Aware attention using PyG interface
    accept Batch data given by PyG

    Args:
    ----------
    embed_dim (int):        the embeding dimension
    num_heads (int):        number of attention heads (default: 8)
    dropout (float):        dropout value (default: 0.0)
    bias (bool):            whether layers have an additive bias (default: False)
    symmetric (bool):       whether K=Q in dot-product attention (default: False)
    gnn_type (str):         GNN type to use in structure extractor (for 'gnn' or 'khopgnn' se).
    se (str):               type of structure extractor ("gnn", "khopgnn", "gign")
    k_hop (int):            number of base GNN layers or the K hop size for khopgnn structure extractor (default=2).
    num_gign_layers (int):  number of HIL layers if se="gign" (default: 3)
    """

    def __init__(self, embed_dim, num_heads=8, dropout=0., bias=False,
        symmetric=False, gnn_type="gcn", se="gnn", k_hop=2, **kwargs): # Added num_gign_layers

        super().__init__(node_dim=0, aggr='add') # node_dim=0 for heterogeneous graphs, or specify if homogeneous
        self.embed_dim = embed_dim
        self.bias = bias
        head_dim = embed_dim // num_heads
        assert head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.num_heads = num_heads
        self.scale = head_dim ** -0.5

        self.se = se
        self.gnn_type = gnn_type # Used if se is 'gnn' or 'khopgnn'

        if self.se == "khopgnn":
            self.khop_structure_extractor = KHopStructureExtractor(embed_dim, gnn_type=gnn_type,
                                                          num_layers=k_hop, **kwargs)
        elif self.se == "gign":
            num_gign_layers = kwargs.get('num_gign_layers', 3)
            self.gign_structure_extractor = GIGNStructureExtractor(
                d_model=embed_dim, # GIGN SE operates on embed_dim (d_model) features
                num_gign_layers=num_gign_layers,
                gign_dropout=kwargs.get('gign_dropout', 0.1)
            )
        elif self.se == "gnn": # Default 'gnn'
            self.structure_extractor = StructureExtractor(embed_dim, gnn_type=gnn_type,
                                                          num_layers=k_hop, **kwargs)
        else:
            raise ValueError(f"Unsupported structure extractor type: {self.se}")
            
        self.attend = nn.Softmax(dim=-1)

        self.symmetric = symmetric
        if symmetric:
            self.to_qk = nn.Linear(embed_dim, embed_dim, bias=bias)
        else:
            self.to_qk = nn.Linear(embed_dim, embed_dim * 2, bias=bias)
        self.to_v = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.attn_dropout = nn.Dropout(dropout)

        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self._reset_parameters()

        self.attn_sum = None # This seems unused, can be removed if confirmed
        self._attn = None # For returning attention scores

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.to_qk.weight)
        nn.init.xavier_uniform_(self.to_v.weight)

        if self.bias:
            nn.init.constant_(self.to_qk.bias, 0.)
            nn.init.constant_(self.to_v.bias, 0.)

    def forward(self,
            x, # (N, d_model) - node features after initial embedding
            edge_index, # Combined edge_index, primarily for 'gnn' or 'khopgnn' SE, or if attention itself uses it
            complete_edge_index, # For full attention computation
            # Args for KHop SE
            subgraph_node_index=None,
            subgraph_edge_index=None,
            subgraph_indicator_index=None,
            subgraph_edge_attr=None, # These are embedded edge attributes for subgraphs
            # Args for GNN SE or main attention
            edge_attr=None, # These are embedded edge attributes for main graph
            ptr=None, # For batching, esp. if 'cls' token and padding are used
            # Args for GIGN SE
            edge_index_intra=None,
            edge_index_inter=None,
            pos=None,
            return_attn=False):
        """
        Compute attention layer.
        """
        # Compute value matrix (from original x, which is d_model)
        v = self.to_v(x) # v: (N, d_model)

        # Compute structure-aware node embeddings (x_struct)
        if self.se == 'khopgnn':
            x_struct = self.khop_structure_extractor(
                x=x, # (N, d_model)
                edge_index=edge_index, # Original graph edges
                subgraph_edge_index=subgraph_edge_index,
                subgraph_indicator_index=subgraph_indicator_index,
                subgraph_node_index=subgraph_node_index,
                subgraph_edge_attr=subgraph_edge_attr, # Embedded subgraph edge attributes
            )
        elif self.se == 'gign':
            if edge_index_intra is None or edge_index_inter is None or pos is None:
                raise ValueError("GIGN SE requires edge_index_intra, edge_index_inter, and pos.")
            x_struct = self.gign_structure_extractor(
                x=x, # (N, d_model)
                edge_index_intra=edge_index_intra,
                edge_index_inter=edge_index_inter,
                pos=pos
            )
        elif self.se == 'gnn': # 'gnn'
            x_struct = self.structure_extractor(x, edge_index, edge_attr=edge_attr) # Pass embedded main edge_attr
        else: # Should not happen due to init check
            x_struct = x 

        # x_struct is now (N, d_model)

        # Compute query and key matrices from x_struct
        if self.symmetric:
            qk_features = self.to_qk(x_struct) # (N, d_model)
            qk = (qk_features, qk_features)
        else:
            qk_features = self.to_qk(x_struct) # (N, 2 * d_model)
            qk = qk_features.chunk(2, dim=-1) # tuple of two (N, d_model) tensors (q, k)
        
        # Compute complete self-attention
        attn_scores_sparse = None # For returning attention if needed

        # self.propagate is used for sparse attention on `complete_edge_index`
        # self.self_attn is used for dense attention (if complete_edge_index is None, implies full graph per batch item)
        if complete_edge_index is not None:
            # This part computes attention messages over the `complete_edge_index` (all pairs)
            # qk_i and qk_j in `message` will be derived from `qk` based on `complete_edge_index`
            # v_j in `message` will be derived from `v`
            out = self.propagate(complete_edge_index, v=v, qk=qk, edge_attr=None, size=None, # edge_attr for attention bias, usually None for TFs
                                 return_attn=return_attn) # Pass return_attn flag
            if return_attn and self._attn is not None:
                # self._attn stores softmaxed attention scores for edges in complete_edge_index
                attn_scores_sparse = torch.sparse_coo_tensor(
                    complete_edge_index,
                    self._attn, # Assuming _attn has shape (num_edges_in_complete_graph, num_heads)
                    size=(x.size(0), x.size(0), self.num_heads) # Check shape carefully
                ).to_dense() # Converts to (N, N, num_heads)
                # Or, if _attn is (num_edges_in_complete_graph), then (N,N)
                # Typically, attention scores are per head. Let's assume self._attn stores flat scores that need reshaping or careful interpretation.
                # The original implementation had .transpose(0,1) - adjust based on _attn's actual structure.
                # For now, let's assume _attn gives a flat list of scores for edges in complete_edge_index
                self._attn = None # Clear stored attention scores

            out = rearrange(out, 'n (h d) -> n h d') # Rearrange from (N, embed_dim) to (N, num_heads, head_dim)
            out = rearrange(out, 'n h d -> n (h d)') # Back to (N, embed_dim)
        else:
            # Fallback to dense self-attention if complete_edge_index is not provided
            # This requires padding/unpadding based on `ptr`
            out, attn_scores_dense_padded = self.self_attn_dense(qk, v, ptr, return_attn=return_attn)
            if return_attn:
                 attn_scores_sparse = attn_scores_dense_padded # This would be (batch_size, num_heads, max_len, max_len)

        return self.out_proj(out), attn_scores_sparse # Return projected output and attention scores

    def message(self, v_j, qk_i, qk_j, edge_attr, index, ptr, size_i, return_attn): # Added return_attn
        """Self-attention message passing for sparse attention (using complete_edge_index)."""
        # qk_i, qk_j are q_target_node, k_source_node from the qk tuple, already (num_edges_in_batch, d_model)
        # v_j is v_source_node, already (num_edges_in_batch, d_model)

        q_i_reshaped = rearrange(qk_i, 'e (h d) -> e h d', h=self.num_heads) # e: num_edges in complete_edge_index
        k_j_reshaped = rearrange(qk_j, 'e (h d) -> e h d', h=self.num_heads)
        v_j_reshaped = rearrange(v_j, 'e (h d) -> e h d', h=self.num_heads)

        # Attention score for each edge, for each head
        attn_score_logits = (q_i_reshaped * k_j_reshaped).sum(dim=-1) * self.scale # (e, num_heads)
        
        # Add edge_attr bias if provided (e.g. for relative positional encodings in attention)
        # This `edge_attr` is specific to attention mechanism, not general graph edge features.
        # Usually None for standard Transformer, unless TAPE-like biases are used.
        if edge_attr is not None: # edge_attr should be (e, num_heads) or broadcastable
            attn_score_logits = attn_score_logits + edge_attr

        # Softmax over incoming edges for each node (index is target node index)
        # utils.softmax takes (value_to_softmax, index_for_grouping, ptr_for_batch (optional), num_nodes_total)
        alpha = utils.softmax(attn_score_logits, index, ptr, size_i) # alpha: (e, num_heads)

        if return_attn: # Store the normalized attention scores (alpha)
            self._attn = alpha 
        
        alpha_dropout = self.attn_dropout(alpha) # (e, num_heads)

        # Weighted sum of values: (e, h, d) * (e, h, 1) -> (e, h, d)
        # Then aggregate by MessagePassing's 'add' aggregator to target nodes.
        # Output of message should be what's aggregated. So, weighted v_j.
        # MessagePassing expects output of message to be `aggr_dim`, which is `embed_dim` here.
        # So, (v_j_reshaped * alpha_dropout.unsqueeze(-1)) is (e, h, d)
        # This needs to be (e, embed_dim) for PyG aggregation.
        msg_output = v_j_reshaped * alpha_dropout.unsqueeze(-1) # (e, num_heads, head_dim)
        return rearrange(msg_output, 'e h d -> e (h d)') # (e, embed_dim)


    def self_attn_dense(self, qk_tuple, v_nodes, ptr, return_attn=False):
        """ Dense Self attention for situations where complete_edge_index is not used.
            Requires padding and unpadding if running in batch mode with `ptr`.
        """
        q_nodes, k_nodes = qk_tuple # Each is (N, d_model)

        # Pad for batched dense attention
        # qk is a tuple of (q, k) where q and k are [N, D_model_QueryKey]
        q_padded, mask = pad_batch(q_nodes, ptr, return_mask=True) # (B, max_len, D_model_QueryKey), (B, max_len)
        k_padded = pad_batch(k_nodes, ptr)                         # (B, max_len, D_model_QueryKey)
        v_padded = pad_batch(v_nodes, ptr)                         # (B, max_len, D_model_Value)

        B, L, _ = q_padded.shape # Batch size, Max Length

        q_reshaped = rearrange(q_padded, 'b l (h d) -> b h l d', h=self.num_heads) # (B, n_heads, L, D_head)
        k_reshaped = rearrange(k_padded, 'b l (h d) -> b h l d', h=self.num_heads) # (B, n_heads, L, D_head)
        v_reshaped = rearrange(v_padded, 'b l (h d) -> b h l d', h=self.num_heads) # (B, n_heads, L, D_head_v)

        # Scaled Dot-Product Attention
        # (B, n_heads, L, D_head) @ (B, n_heads, D_head, L) -> (B, n_heads, L, L)
        dots = torch.matmul(q_reshaped, k_reshaped.transpose(-1, -2)) * self.scale

        # Apply mask (mask out padding positions)
        # Mask is (B, L), needs to be (B, 1, 1, L) for columns and (B, 1, L, 1) for rows
        if mask is not None:
            # mask.unsqueeze(1).unsqueeze(2) makes it (B, 1, 1, L) - masks columns (keys)
            # mask.unsqueeze(1).unsqueeze(-1) makes it (B, 1, L, 1) - masks rows (queries)
            # We need to mask positions that don't exist.
            # A value of False in the mask means it's a padding position.
            # Mask should be (B, L). For attention (B,H,L,L), mask needs to be (B,1,L,L) or (B,1,1,L)
            # Effectively, if mask[b,j] is False, then attn[b,:,i,j] should be -inf.
            # And if mask[b,i] is False, then attn[b,:,i,j] should be -inf.
            # So, combined_mask[b,i,j] = mask[b,i] AND mask[b,j]
            attention_mask = mask.unsqueeze(1) * mask.unsqueeze(2) # (B, L, L)
            attention_mask = attention_mask.unsqueeze(1) # (B, 1, L, L) for broadcasting over heads
            dots = dots.masked_fill(~attention_mask, float('-inf'))


        attn_probs = self.attend(dots) # Softmax over last dim (keys for each query) -> (B, n_heads, L, L)
        attn_probs = self.attn_dropout(attn_probs)

        # Apply attention to values
        # (B, n_heads, L, L) @ (B, n_heads, L, D_head_v) -> (B, n_heads, L, D_head_v)
        out_padded = torch.matmul(attn_probs, v_reshaped)
        
        # Concatenate heads and unpad
        out_padded = rearrange(out_padded, 'b h l d -> b l (h d)') # (B, L, D_model_Value)
        out_unpadded = unpad_batch(out_padded, ptr) # (N, D_model_Value)

        if return_attn:
            return out_unpadded, attn_probs # Return unpadded output and (padded) attention probabilities
        return out_unpadded, None


class StructureExtractor(nn.Module):
    r""" K-subtree structure extractor. Computes the structure-aware node embeddings using the
    k-hop subtree centered around each node.

    Args:
    ----------
    embed_dim (int):        the embeding dimension
    gnn_type (str):         GNN type to use in structure extractor. (gcn, gin, pna, etc)
    num_layers (int):       number of GNN layers
    batch_norm (bool):      apply batch normalization or not
    concat (bool):          whether to concatenate the initial edge features
    khopgnn (bool):         whether to use the subgraph instead of subtree
    """

    def __init__(self, embed_dim, gnn_type="gcn", num_layers=3,
                 batch_norm=True, concat=True, khopgnn=False, **kwargs):
        super().__init__()
        self.num_layers = num_layers
        self.khopgnn = khopgnn # This flag is a bit confusingly named here.
                              # If True, it means this SE is *part of* KHopStructureExtractor,
                              # and its output will be aggregated by scatter_add/mean.
        self.concat = concat
        self.gnn_type = gnn_type
        layers = []
        # kwargs might include 'edge_dim' if the GNN layers use edge features
        self.edge_dim = kwargs.get('edge_dim', None)

        for _ in range(num_layers):
            # Pass edge_dim to GNN layer constructor if GNN type supports it
            layer_kwargs = kwargs.copy()
            # Ensure only relevant GNN layers get edge_dim if it's not None
            if self.gnn_type in EDGE_GNN_TYPES and self.edge_dim is not None:
                layer_kwargs['edge_dim'] = self.edge_dim
            else: # For GNNs not using edge_dim, don't pass it, or pass None explicitly
                layer_kwargs.pop('edge_dim', None)

            layers.append(get_simple_gnn_layer(gnn_type, embed_dim, **layer_kwargs))
        self.gcn = nn.ModuleList(layers)

        self.relu = nn.ReLU()
        self.batch_norm = batch_norm
        
        # Input to out_proj is sum of embeddings if concat=True
        inner_dim = (num_layers + 1) * embed_dim if concat else embed_dim

        if batch_norm and not self.khopgnn: # BN applied before out_proj if not part of KHopSE
             # For KHopSE, BN is typically applied *after* cat with original features.
            self.bn = nn.BatchNorm1d(inner_dim)

        self.out_proj = nn.Linear(inner_dim, embed_dim)

    def forward(self, x, edge_index, edge_attr=None, # edge_attr are embedded edge features
            subgraph_indicator_index=None, agg="sum"):
        
        x_orig = x
        x_cat = [x_orig] if self.concat else [] # Start with original x if concat

        # GNN layers
        h = x_orig
        for gcn_layer in self.gcn:
            if self.gnn_type in EDGE_GNN_TYPES and self.edge_dim is not None:
                if edge_attr is None: # Should not happen if edge_dim is provided
                    raise ValueError(f"GNN type {self.gnn_type} expects edge_attr but None provided.")
                h = self.relu(gcn_layer(h, edge_index, edge_attr=edge_attr))
            else:
                h = self.relu(gcn_layer(h, edge_index))
            
            if self.concat:
                x_cat.append(h)
        
        if self.concat:
            processed_x = torch.cat(x_cat, dim=-1) # (N, inner_dim)
        else: # Only use last layer's output
            processed_x = h # (N, embed_dim)


        if self.khopgnn: # This means it's used within KHopStructureExtractor
            # Aggregation happens here if it's the inner part of KHopSE
            if subgraph_indicator_index is None:
                 raise ValueError("subgraph_indicator_index needed for khopgnn=True in StructureExtractor")
            if agg == "sum":
                aggregated_x = scatter_add(processed_x, subgraph_indicator_index, dim=0)
            elif agg == "mean":
                aggregated_x = scatter_mean(processed_x, subgraph_indicator_index, dim=0)
            else:
                raise ValueError(f"Unknown aggregation type: {agg}")
            # No BN or final projection here if it's khopgnn part; that's done in KHopSE itself.
            return aggregated_x # (num_original_nodes, inner_dim or embed_dim)

        # If standalone StructureExtractor (not part of KHopSE)
        if self.num_layers > 0 and self.batch_norm:
            processed_x = self.bn(processed_x)

        final_x = self.out_proj(processed_x) # Project to embed_dim
        return final_x


class KHopStructureExtractor(nn.Module):
    r""" K-subgraph structure extractor. Extracts a k-hop subgraph centered around
    each node and uses a GNN on each subgraph to compute updated structure-aware
    embeddings.

    Args:
    ----------
    embed_dim (int):        the embeding dimension
    gnn_type (str):         GNN type to use in structure extractor. (gcn, gin, pna, etc)
    num_layers (int):       number of GNN layers for the GNN operating on subgraphs
    concat (bool):          whether to concatenate the initial edge features (for the inner GNN) - typically False
    khopgnn (bool):         (internal flag, should be True for this class's structure_extractor)
    """
    def __init__(self, embed_dim, gnn_type="gcn", num_layers=3, batch_norm=True,
            concat=False, khopgnn=True, **kwargs): # concat=False is typical for inner GNN of KHopSE
        super().__init__()
        self.num_layers = num_layers # GNN layers for subgraph processing
        # self.khopgnn = khopgnn # This class IS the KHop SE.

        self.batch_norm = batch_norm

        # The inner StructureExtractor processes the disjoint union of all k-hop subgraphs.
        # It should not concat features internally, as its output (embed_dim)
        # will be concatenated with the original node features.
        # `khopgnn=True` tells StructureExtractor to do scatter_add/mean.
        self.edge_dim_for_subgraphs = kwargs.get('edge_dim', None) # If subgraphs have edge features

        structure_extractor_kwargs = kwargs.copy()
        if self.edge_dim_for_subgraphs is not None and gnn_type in EDGE_GNN_TYPES:
            structure_extractor_kwargs['edge_dim'] = self.edge_dim_for_subgraphs
        else:
            structure_extractor_kwargs.pop('edge_dim', None)


        self.structure_extractor = StructureExtractor(
            embed_dim, # Operates on embed_dim features
            gnn_type=gnn_type,
            num_layers=num_layers, # Layers for subgraph GNN
            concat=concat, # concat for internal GNN layers (usually False or True based on preference)
            khopgnn=True,  # Critical: tells StructureExtractor to aggregate subgraph features
            **structure_extractor_kwargs # Pass edge_dim for subgraphs etc.
        )
        
        # Output of structure_extractor will be embed_dim (if concat=False in SE) or (num_layers+1)*embed_dim (if concat=True in SE)
        # Let's assume structure_extractor (with khopgnn=True) outputs embed_dim for simplicity here.
        # If SE's concat=True, its output is (num_layers+1)*embed_dim. Then Linear layer below needs adjustment.
        # For now, assume SE (khopgnn=True path) outputs `embed_dim` after its internal projection if concat was True.
        # Or more simply, ensure SE's `concat` is False.

        # Features from original node (embed_dim) + features from its subgraph (embed_dim from SE)
        feat_dim_after_cat = embed_dim + embed_dim # If SE output is embed_dim

        if batch_norm:
            self.bn = nn.BatchNorm1d(feat_dim_after_cat)

        self.out_proj = nn.Linear(feat_dim_after_cat, embed_dim)

    def forward(self, x, edge_index, subgraph_edge_index, # x is (N, embed_dim)
                subgraph_node_index, subgraph_indicator_index, # Used by inner SE
                edge_attr=None, # Main graph edge_attr, not used directly here but by Attention's other SEs
                subgraph_edge_attr=None # Embedded edge attributes for subgraphs
                ):

        # x_subgraph_nodes: features of nodes participating in any k-hop subgraph
        # These are (N_subgraph_nodes, embed_dim) where N_subgraph_nodes is sum of nodes in all subgraphs
        x_subgraph_nodes = x[subgraph_node_index] # Select features for subgraph nodes

        # x_struct_aggregated: (N, embed_dim), result of GNN on subgraphs, aggregated back to original N nodes
        x_struct_aggregated = self.structure_extractor(
            x=x_subgraph_nodes,
            edge_index=subgraph_edge_index, # Edges of the disjoint k-hop subgraphs
            edge_attr=subgraph_edge_attr,   # Embedded edge features for these subgraph edges
            subgraph_indicator_index=subgraph_indicator_index, # To map results back to N original nodes
            agg="sum", # Aggregation method (sum or mean)
        )
        
        # Concatenate original node features with their aggregated k-hop subgraph representations
        x_combined = torch.cat([x, x_struct_aggregated], dim=-1) # (N, 2 * embed_dim)
        
        if self.batch_norm:
            x_combined = self.bn(x_combined)
        
        x_projected = self.out_proj(x_combined) # Project back to (N, embed_dim)

        return x_projected


class TransformerEncoderLayer(nn.TransformerEncoderLayer):
    r"""Structure-Aware Transformer layer, made up of structure-aware self-attention and feed-forward network.

    Args:
    ----------
        d_model (int):      the number of expected features in the input (required).
        nhead (int):        the number of heads in the multiheadattention models (default=8).
        dim_feedforward (int): the dimension of the feedforward network model (default=512).
        dropout:            the dropout value (default=0.1).
        activation:         the activation function of the intermediate layer, can be a string
            ("relu" or "gelu") or a unary callable (default: relu).
        batch_norm:         use batch normalization instead of layer normalization (default: True).
        pre_norm:           pre-normalization or post-normalization (default=False).
        gnn_type:           base GNN model to extract subgraph representations.
                            One can implememnt customized GNN in gnn_layers.py (default: gcn).
        se:                 structure extractor to use, either gnn or khopgnn or gign (default: gnn).
        k_hop:              the number of base GNN layers or the K hop size for khopgnn structure extractor (default=2).
    """
    def __init__(self, d_model, nhead=8, dim_feedforward=512, dropout=0.1,
                activation="relu", batch_norm=True, pre_norm=False,
                gnn_type="gcn", se="gnn", k_hop=2, **kwargs): # Propagate kwargs
        super().__init__(d_model, nhead, dim_feedforward, dropout, activation=activation) # Pass activation string

        # **kwargs may include edge_dim for GNN/KHop SE, num_gign_layers for GIGN SE
        self.self_attn = Attention(d_model, nhead, dropout=dropout,
            bias=False, gnn_type=gnn_type, se=se, k_hop=k_hop, **kwargs) # Pass kwargs
        
        self.batch_norm = batch_norm
        self.pre_norm = pre_norm
        if batch_norm: # Use BatchNorm instead of LayerNorm
            self.norm1 = nn.BatchNorm1d(d_model)
            self.norm2 = nn.BatchNorm1d(d_model)
        # Else, super().__init__ already created self.norm1 and self.norm2 as LayerNorm

    def forward(self, x, edge_index, complete_edge_index,
            subgraph_node_index=None, subgraph_edge_index=None,
            subgraph_edge_attr=None, # Embedded subgraph edge_attr
            subgraph_indicator_index=None,
            edge_attr=None, # Embedded main graph edge_attr
            degree=None, ptr=None,
            # GIGN specific data
            edge_index_intra=None, edge_index_inter=None, pos=None,
            return_attn=False,
        ):

        # Pre-normalization block
        if self.pre_norm:
            # If using LayerNorm (batch_norm=False), self.norm1(x) is correct.
            # If using BatchNorm1d (batch_norm=True), input needs to be (N, C) or (B, C, L).
            # PyG graphs are usually (N, C), so BatchNorm1d(d_model) should work.
            x_norm1 = self.norm1(x)
        else: # Post-normalization, apply norm after residual
            x_norm1 = x

        # Self-attention block
        # x2 is \Delta x from self-attention
        x2, attn = self.self_attn(
            x_norm1, # Input to attention is pre-normed x
            edge_index,
            complete_edge_index,
            # KHOP SE args
            subgraph_node_index=subgraph_node_index,
            subgraph_edge_index=subgraph_edge_index,
            subgraph_indicator_index=subgraph_indicator_index,
            subgraph_edge_attr=subgraph_edge_attr, # Embedded subgraph edge features
            # GNN SE args
            edge_attr=edge_attr, # Embedded main graph edge features
            ptr=ptr,
            # GIGN SE args
            edge_index_intra=edge_index_intra,
            edge_index_inter=edge_index_inter,
            pos=pos,
            return_attn=return_attn
        )

        if degree is not None and self.se == "khopgnn": # Degree scaling, if applicable (orig had this for khopgnn)
             # Check if this degree scaling is always desired or specific to some setup
            x2 = degree.unsqueeze(-1) * x2
        
        x = x + self.dropout1(x2) # Add residual connection (x from before norm1 if pre_norm)

        if not self.pre_norm: # Post-normalization for first block
            x = self.norm1(x)

        # Feed-forward block
        if self.pre_norm:
            x_norm2 = self.norm2(x) # Pre-norm for FFN
        else: # Post-normalization, apply norm after FFN residual
            x_norm2 = x

        # Original Transformer FFN: linear1 -> activation -> dropout -> linear2
        # super().linear1, super().linear2, super().dropout, super().activation exist from nn.TransformerEncoderLayer
        x_ffn_out = self.linear2(self.dropout(self.activation(self.linear1(x_norm2))))
        x = x + self.dropout2(x_ffn_out) # Add residual (x from after first block if pre_norm)

        if not self.pre_norm: # Post-normalization for second block
            x = self.norm2(x)
            
        # `attn` here is for the current layer. If `return_attn` is True for the whole Encoder,
        # it typically means returning attentions from all layers or the last one.
        # This layer returns its own `attn`. The GraphTransformerEncoder handles aggregation if needed.
        if return_attn:
            return x, attn # Return processed x and attention scores from this layer
        return x # Default, only return processed x