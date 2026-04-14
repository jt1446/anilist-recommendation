print("starting backend...")

from modules.gnn_model import GNNEncoder
from modules.trainer import main as train_models
from modules.utils import load_processed_data, extract_embeddings
from modules.visualizer import create_latent_space_map
import torch
import os
import pandas as pd

# Entry point for running the training pipeline
if __name__ == "__main__":
    print("Starting training pipeline...")
    train_models()

    print("\n--- Generating Latent Space Visualization ---")
    try:
        # Load weights and data
        weights_path = 'trained_weights.pth'
        checkpoint = torch.load(weights_path, map_location='cpu', weights_only=False)
        
        # Load data used for mapping
        print("Loading data...")
        metadata_path = 'data/anime_metadata.csv'
        interactions_path = 'data/interactions.csv'
        
        # We need the same processing used during training to get the correct dimensions
        from modules.data_processor import build_feature_matrix, create_interaction_graph
        features, anime_id_to_idx = build_feature_matrix(metadata_path)
        
        # Load weights
        print(f"Loading weights from {weights_path}...")
        checkpoint = torch.load(weights_path, map_location='cpu', weights_only=False)
        
        # Check which checkpoint keys exist and use the GNN state dict
        if 'gnn_state_dict' in checkpoint:
            state_dict = checkpoint['gnn_state_dict']
            gnn_config = checkpoint.get('gnn_config', {})
            input_dim = gnn_config.get('input_dim', features.shape[1])
            hidden_dim = gnn_config.get('hidden_dim', 128)
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            gnn_config = checkpoint.get('gnn_config', {})
            input_dim = gnn_config.get('input_dim', features.shape[1])
            hidden_dim = gnn_config.get('hidden_dim', 128)
        else:
            state_dict = checkpoint
            input_dim = features.shape[1]
            hidden_dim = 128
            
        # Replicate edge_index exactly as done in trainer.py
        print("Replicating graph structure...")
        edge_index, user_id_to_idx = create_interaction_graph(interactions_path, anime_id_to_idx)
        
        num_users = len(user_id_to_idx)
        num_items = features.shape[0]

        # Load user embedding weights if available, otherwise fall back to zeros.
        if 'user_embedding_state_dict' in checkpoint:
            user_embedding = torch.nn.Embedding(num_users, input_dim)
            user_embedding.load_state_dict(checkpoint['user_embedding_state_dict'])
            user_features = user_embedding.weight
        else:
            user_features = torch.zeros((num_users, input_dim))

        full_features = torch.cat([user_features, features], dim=0)

        print(f"Graph ready: {full_features.shape[0]} total nodes ({num_items} items), {edge_index.shape[1]} edges.")
        
        # Initialize model
        model = GNNEncoder(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=hidden_dim)
        
        # Load weights into model
        model.load_state_dict(state_dict)

        # Extract embeddings
        print("Extracting embeddings...")
        all_embeddings = extract_embeddings(
            model=model,
            feature_matrix=full_features,
            edge_index=edge_index
        )
        
        # We only want to visualize the items (anime), which are after users in the bipartite graph
        item_embeddings = all_embeddings[num_users:]
        
        # Load metadata for visualization labels
        metadata_df = pd.read_csv(metadata_path)
        
        # Visualize
        print("Generating visualization...")
        fig = create_latent_space_map(item_embeddings, metadata_df)
        fig.write_html("latent_space.html")
        print("Success! Created latent_space.html")
        
    except Exception as e:
        print(f"Visualization generation failed: {e}")
