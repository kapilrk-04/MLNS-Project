# --- START OF FILE gin_regression_model.py ---
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, GINEConv, global_mean_pool, global_add_pool

class GINRegression(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels=1, num_layers=3,
                 dropout=0.1, use_edge_features=False, edge_dim=None, global_pool_type='mean',
                 train_eps=False): # train_eps is a GIN-specific parameter
        super().__init__()
        self.use_edge_features = use_edge_features
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        current_channels = in_channels
        for i in range(num_layers):
            # Define the MLP for GIN/GINE layers
            # Typically a 2-layer MLP: Linear -> ReLU -> Linear
            mlp = nn.Sequential(
                nn.Linear(current_channels, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels)
            )

            if use_edge_features and edge_dim is not None:
                # GINEConv incorporates edge features
                conv = GINEConv(mlp, train_eps=train_eps, edge_dim=edge_dim)
            else:
                # GINConv does not use edge features explicitly in its formulation
                conv = GINConv(mlp, train_eps=train_eps)

            self.convs.append(conv)
            self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
            current_channels = hidden_channels # For the next layer's MLP input

        self.dropout = dropout

        if global_pool_type == 'mean':
            self.global_pool = global_mean_pool
        elif global_pool_type == 'add':
            self.global_pool = global_add_pool
        else:
            raise ValueError(f"Unsupported global_pool_type: {global_pool_type}")

        # Regression head
        self.fc1 = nn.Linear(hidden_channels, hidden_channels // 2)
        self.fc2 = nn.Linear(hidden_channels // 2, out_channels)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_attr = data.edge_attr if hasattr(data, 'edge_attr') and self.use_edge_features else None

        for i in range(len(self.convs)):
            if self.use_edge_features and edge_attr is not None:
                x = self.convs[i](x, edge_index, edge_attr=edge_attr)
            else:
                x = self.convs[i](x, edge_index) # GINConv
            x = self.batch_norms[i](x)
            x = F.relu(x) # Apply activation after BN
            if i < len(self.convs) - 1: # No dropout after the last GIN layer's features
                x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.global_pool(x, batch)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training) # Dropout before final layer
        x = self.fc2(x)
        return x.squeeze(-1) # Ensure output is [batch_size]

if __name__ == '__main__':
    # Example instantiation
    node_feat_dim = 35
    edge_feat_dim = 2 # Example: one-hot for intra/inter

    model_gin = GINRegression(in_channels=node_feat_dim, hidden_channels=64, num_layers=3,
                              use_edge_features=False) # Standard GIN
    print("GIN Model (no edge features):\n", model_gin)

    model_gine = GINRegression(in_channels=node_feat_dim, hidden_channels=64, num_layers=3,
                               use_edge_features=True, edge_dim=edge_feat_dim) # GINE
    print("\nGINE Model (with edge features):\n", model_gine)

    # Dummy data for testing forward pass
    from torch_geometric.data import Data, Batch
    num_nodes_graph1 = 5
    num_nodes_graph2 = 7

    data1 = Data(
        x=torch.randn(num_nodes_graph1, node_feat_dim),
        edge_index=torch.randint(0, num_nodes_graph1, (2, 10)),
        edge_attr=torch.randn(10, edge_feat_dim),
        y=torch.tensor([1.0])
    )
    data2 = Data(
        x=torch.randn(num_nodes_graph2, node_feat_dim),
        edge_index=torch.randint(0, num_nodes_graph2, (2, 12)),
        edge_attr=torch.randn(12, edge_feat_dim),
        y=torch.tensor([2.0])
    )
    batch_data = Batch.from_data_list([data1, data2])

    print("\nTesting GIN model forward pass:")
    output_gin = model_gin(batch_data)
    print("GIN Output shape:", output_gin.shape)

    print("\nTesting GINE model forward pass:")
    output_gine = model_gine(batch_data)
    print("GINE Output shape:", output_gine.shape)
# --- END OF FILE gin_regression_model.py ---