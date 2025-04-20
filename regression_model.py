import torch
from torch import nn
import torch_geometric.nn as gnn
from layers import TransformerEncoderLayer # Corrected import path
from einops import repeat 

class GraphTransformerEncoder(nn.TransformerEncoder):
    def forward(self, x, edge_index, complete_edge_index,
            subgraph_node_index=None, subgraph_edge_index=None,
            subgraph_edge_attr=None, subgraph_indicator_index=None, 
            edge_attr=None, degree=None,
            ptr=None, 
            edge_index_intra=None, edge_index_inter=None, pos=None,
            return_attn=False):
        
        output = x
        attns_list = [] 

        for mod_layer in self.layers: # renamed self.layers to mod_layer to avoid conflict
            # TransformerEncoderLayer expects src, not x
            layer_output = mod_layer(src=output, # Pass current output as src
                edge_index=edge_index, complete_edge_index=complete_edge_index,
                subgraph_node_index=subgraph_node_index, subgraph_edge_index=subgraph_edge_index,
                subgraph_indicator_index=subgraph_indicator_index, subgraph_edge_attr=subgraph_edge_attr,
                edge_attr=edge_attr, degree=degree, ptr=ptr,
                edge_index_intra=edge_index_intra, edge_index_inter=edge_index_inter, pos=pos,
                return_attn=return_attn 
            )
            if return_attn:
                output, layer_attn_scores = layer_output
                attns_list.append(layer_attn_scores) 
            else:
                output = layer_output
        
        if self.norm is not None: 
            output = self.norm(output)
        
        if return_attn:
            return output, attns_list 
        return output


class GraphTransformerRegression(nn.Module):
    def __init__(self, in_size, out_dim=1, d_model=128, num_heads=8,
                 dim_feedforward=512, dropout=0.1, num_layers=4,
                 batch_norm=False, abs_pe=False, abs_pe_dim=0, 
                 gnn_type="gcn", se="gnn", use_edge_attr=False, num_edge_features=0, # Defaults for no edge_attr
                 in_embed=True, edge_embed=False, use_global_pool=True, 
                 global_pool='mean', 
                 **kwargs): 
        super().__init__()

        self.abs_pe_active = abs_pe # Renamed
        if self.abs_pe_active and abs_pe_dim > 0:
            self.embedding_abs_pe = nn.Linear(abs_pe_dim, d_model)

        self.in_embed_active = in_embed # Renamed
        if self.in_embed_active:
            if isinstance(in_size, int) and in_size > 0: # Check in_size > 0 for Embedding
                self.embedding_node = nn.Embedding(in_size, d_model) # Renamed
            else: # Assumes in_size is feature dimension for Linear
                 self.embedding_node = nn.Linear(in_features=in_size, out_features=d_model, bias=False)
        else: 
            self.embedding_node = nn.Identity()
            if in_size != d_model:
                 print(f"Warning: in_embed=False but in_size ({in_size}) != d_model ({d_model}).")

        self.use_edge_attr_active = use_edge_attr # Renamed
        self.embedding_edge = None # Initialize
        if self.use_edge_attr_active and num_edge_features > 0 : 
            edge_dim_internal = kwargs.get('edge_dim', 32) 
            if edge_embed: 
                self.embedding_edge = nn.Embedding(num_edge_features, edge_dim_internal)
            else: 
                 self.embedding_edge = nn.Linear(in_features=num_edge_features, out_features=edge_dim_internal, bias=False)
            kwargs['edge_dim'] = edge_dim_internal 
        else:
            kwargs['edge_dim'] = None 

        # Pass deg to PNA layers if PNA is used in SE
        # This requires 'deg' to be computed per batch and passed to Attention -> SE
        # For now, assume 'deg' is handled if PNA is selected and data provides it.
        # kwargs['deg'] = data.degree_for_pna if hasattr(data, 'degree_for_pna') else None
        # This is complex to handle globally, usually PNAConv takes `deg` at initialization.
        # The `get_simple_gnn_layer` needs `deg` in kwargs if PNA is used.
        # The `train_regression...` script should prepare `deg` if PNA is used.
        # For simplicity, if using PNA, ensure `deg` is part of `**kwargs` passed to model.

        self.current_se_type = se # Renamed
        encoder_layer_instance = TransformerEncoderLayer( # Renamed
            d_model, num_heads, dim_feedforward, dropout, batch_norm=batch_norm,
            gnn_type=gnn_type, se=self.current_se_type, **kwargs) 
            
        self.encoder_module = GraphTransformerEncoder(encoder_layer_instance, num_layers, # Renamed
                                         norm=nn.LayerNorm(d_model) if not batch_norm and num_layers > 0 else None)

        self.global_pool_method = global_pool # Renamed
        self.use_global_pool_active = use_global_pool # Renamed
        if self.use_global_pool_active:
            if self.global_pool_method == 'mean':
                self.pooling_fn = gnn.global_mean_pool # Renamed
            elif self.global_pool_method == 'add':
                self.pooling_fn = gnn.global_add_pool
            elif self.global_pool_method == 'cls':
                self.cls_token_param = nn.Parameter(torch.randn(1, d_model)) # Renamed
                self.pooling_fn = None 
            else:
                 raise ValueError(f"Unsupported global pooling: {self.global_pool_method}")
        else:
            self.pooling_fn = None

        self.regressor_head_fc = nn.Sequential( # Renamed
            nn.Linear(d_model, d_model // 2), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(d_model // 2, out_dim) 
        )

    def forward(self, data, return_attn=False):
        x_in, batch_indices = data.x, data.batch # Renamed
        edge_index_main = data.edge_index if hasattr(data, 'edge_index') else None
        edge_attr_main_raw = data.edge_attr if hasattr(data, 'edge_attr') else None 

        subgraph_node_idx_khop, subgraph_edge_idx_khop, subgraph_indicator_idx_khop, subgraph_edge_attr_raw_khop = None, None, None, None
        if self.current_se_type == "khopgnn":
            subgraph_node_idx_khop = data.subgraph_node_idx if hasattr(data, "subgraph_node_idx") else None
            subgraph_edge_idx_khop = data.subgraph_edge_index if hasattr(data, "subgraph_edge_index") else None
            subgraph_indicator_idx_khop = data.subgraph_indicator if hasattr(data, "subgraph_indicator") else None
            subgraph_edge_attr_raw_khop = data.subgraph_edge_attr if hasattr(data, "subgraph_edge_attr") else None
            if subgraph_node_idx_khop is None : raise ValueError("Missing KHopGNN attributes for se='khopgnn'")

        edge_idx_intra_gign, edge_idx_inter_gign, pos_gign = None, None, None
        if self.current_se_type == "gign":
            edge_idx_intra_gign = data.edge_index_intra if hasattr(data, 'edge_index_intra') else None
            edge_idx_inter_gign = data.edge_index_inter if hasattr(data, 'edge_index_inter') else None
            pos_gign = data.pos if hasattr(data, 'pos') else None
            if edge_idx_intra_gign is None or pos_gign is None : # inter can be empty
                raise ValueError("GIGN SE requires at least edge_index_intra and pos.")
            
        complete_edge_idx_tf = data.complete_edge_index if hasattr(data, 'complete_edge_index') else None
        abs_pe_features_raw = data.abs_pe if hasattr(data, 'abs_pe') else None 
        degree_info = data.degree if hasattr(data, 'degree') else None

        h_feat = self.embedding_node(x_in) # Renamed

        if self.abs_pe_active and abs_pe_features_raw is not None:
            abs_pe_embedded = self.embedding_abs_pe(abs_pe_features_raw) 
            h_feat = h_feat + abs_pe_embedded

        edge_attr_main_emb = None
        if self.use_edge_attr_active and edge_attr_main_raw is not None and self.embedding_edge is not None:
            if edge_attr_main_raw.numel() > 0:
                 edge_attr_main_emb = self.embedding_edge(edge_attr_main_raw) 
        
        subgraph_edge_attr_khop_emb = None
        if self.current_se_type == "khopgnn" and subgraph_edge_attr_raw_khop is not None and self.embedding_edge is not None:
             if subgraph_edge_attr_raw_khop.numel() > 0:
                subgraph_edge_attr_khop_emb = self.embedding_edge(subgraph_edge_attr_raw_khop)

        current_batch_ptr = data.ptr if hasattr(data, 'ptr') else None

        if self.global_pool_method == 'cls' and self.use_global_pool_active:
            num_graphs = data.num_graphs 
            cls_tokens_batch = repeat(self.cls_token_param, '() d -> b d', b=num_graphs) 
            h_feat = torch.cat([h_feat, cls_tokens_batch], dim=0) 
            cls_batch_map = torch.arange(num_graphs, device=batch_indices.device)
            batch_indices = torch.cat([batch_indices, cls_batch_map], dim=0)
            if current_batch_ptr is not None: # Update ptr if using cls token with padding logic
                # This assumes cls tokens are added one per graph, at the end of each graph's nodes.
                # Or, if global_add_pool etc is used, ptr needs to reflect the new total node count.
                # If CLS tokens are at the very end (after all real nodes), ptr needs careful handling
                # for `pad_batch` if it's used by dense attention.
                # The current `GraphTransformerEncoderLayer` implementation uses `ptr` for `pad_batch`
                # and for `softmax` in sparse attention.
                # If CLS tokens are simply appended, ptr needs to have one more entry,
                # and last element of ptr should be new_total_nodes.
                # This gets complex. For now, relying on global_mean_pool/add_pool or careful CLS selection.
                # If CLS tokens are appended like this: total_nodes_orig = x_in.size(0)
                # new_ptr = torch.cat([ptr, ptr[-1:] + torch.arange(1, num_graphs + 1).to(ptr.device)], dim=0) - this is wrong.
                # ptr should be updated to account for one extra node per graph if CLS is per graph.
                # If CLS tokens are just appended to the whole batch of nodes:
                # The current ptr still defines boundaries for original graphs. CLS tokens are outside.
                # The pooling logic `output_features[original_num_nodes:]` relies on this.
                pass


        encoder_output_val = self.encoder_module( # Use renamed
            h_feat,
            edge_index=edge_index_main,       
            complete_edge_index=complete_edge_idx_tf,
            subgraph_node_index=subgraph_node_idx_khop,
            subgraph_edge_index=subgraph_edge_idx_khop,
            subgraph_indicator_index=subgraph_indicator_idx_khop,
            subgraph_edge_attr=subgraph_edge_attr_khop_emb, 
            edge_attr=edge_attr_main_emb, 
            degree=degree_info,
            ptr=current_batch_ptr, 
            edge_index_intra=edge_idx_intra_gign,
            edge_index_inter=edge_idx_inter_gign,
            pos=pos_gign,
            return_attn=return_attn
        )

        if return_attn:
            output_node_features, attns_list_out = encoder_output_val # Renamed
        else:
            output_node_features = encoder_output_val
            attns_list_out = None

        if self.use_global_pool_active:
            if self.global_pool_method == 'cls':
                original_num_nodes = x_in.size(0) 
                graph_level_emb = output_node_features[original_num_nodes:] # Renamed
            else:
                graph_level_emb = self.pooling_fn(output_node_features, batch_indices) 
        else: 
            graph_level_emb = output_node_features 

        prediction_out = self.regressor_head_fc(graph_level_emb) # Renamed

        if prediction_out.shape[-1] == 1:
            final_pred = prediction_out.squeeze(-1) 
        else:
            final_pred = prediction_out 
        
        if return_attn:
            return final_pred, attns_list_out
        return final_pred