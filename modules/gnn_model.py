'''
Write the GNN model as a notebook in google colab, then download as a .py and put it here.
Make sure to use the functions found in utils.py for loading data, mapping ids, etc.

Input:

    Node Features: A PyTorch tensor (from the Data Scientist) representing anime metadata (genres, tags, etc.).

    Edge Index: A coordinate format (COO) tensor representing the userId to mediaId connections from interactions.csv.

Output:

    Latent Embeddings: A high-dimensional tensor (e.g., 128-dim) where every mediaId is represented by a vector that captures its "position" in the anime graph.
'''
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

class GNNEncoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=128, dropout=0.2):
        super(GNNEncoder, self).__init__()
        # Store initial projection to ensure it's and stored/reusable
        self.node_features = None
        self.dropout_p = dropout
        # First GraphSAGE layer: aggregates neighbors using mean/pool
        self.conv1 = SAGEConv(input_dim, hidden_dim)
        # Second GraphSAGE layer: produces final latent embeddings
        self.conv2 = SAGEConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        """
        Args:
            x: Node Features tensor [num_nodes, input_dim]
            edge_index: COO format tensor [2, num_edges]
        Returns:
            z: Latent Embeddings [num_nodes, output_dim]
        """
        # First Layer + ReLU
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_p, training=self.training)

        # Second Layer
        z = self.conv2(x, edge_index)
        z = F.dropout(z, p=self.dropout_p, training=self.training)

        return z

def get_spatial_embeddings(model, feature_matrix, edge_index):
    """The forward pass that generates the latest latent vectors for all anime based on current graph structure."""
    model.eval()
    with torch.no_grad():
        embeddings = model(feature_matrix, edge_index)
    return embeddings

# Example Usage (Placeholder for your specific tensors):
# model = AnimeGNN(input_dim=node_features.shape[1])
# embeddings = model(node_features, edge_index)