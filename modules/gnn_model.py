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
from torch_geometric.nn import GCNConv

class AnimeGNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=128):
        super(AnimeGNN, self).__init__()
        # First GCN layer: maps input features to hidden space
        self.conv1 = GCNConv(input_dim, hidden_dim)
        # Second GCN layer: produces final latent embeddings
        self.conv2 = GCNConv(hidden_dim, output_dim)

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
        x = F.dropout(x, p=0.2, training=self.training)

        # Second Layer
        z = self.conv2(x, edge_index)

        return z

# Example Usage (Placeholder for your specific tensors):
# model = AnimeGNN(input_dim=node_features.shape[1])
# embeddings = model(node_features, edge_index)