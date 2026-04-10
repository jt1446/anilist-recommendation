print("starting backend...")

from modules.gnn_model import AnimeGNN
from modules.trainer import main as train_models
from modules.utils import load_processed_data, to_edge_index
from modules.visualizer import extract_embeddings, create_latent_space_map
import torch
import os

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
        result = load_processed_data()
        if isinstance(result, tuple) and len(result) == 3:
            _, metadata_df, interactions_df = result
        else:
            metadata_df, interactions_df = result
            
        # Recreate the exact inputs used in trainer.py
        num_items = len(metadata_df)
        input_dim = checkpoint.get('gnn_config', {}).get('input_dim', 128)
        
        # Replicate feature generation from trainer.py
        if num_items < input_dim:
            node_features = torch.eye(input_dim)[:num_items]
        else:
            # Note: If trainer used random projection, we might want to save it, 
            # but for now we follow the trainer's logic:
            torch.manual_seed(42) # Try to be consistent if possible
            node_features = torch.randn(num_items, input_dim)
            
        edge_index = to_edge_index(interactions_df)

        # Extract
        embeddings = extract_embeddings(
            model_class=AnimeGNN,
            model_path=weights_path,
            num_items=num_items,
            input_dim=input_dim,
            node_features=node_features,
            edge_index=edge_index
        )
        
        # Visualize
        fig = create_latent_space_map(embeddings, metadata_df)
        fig.write_html("latent_space.html")
        print("Success! Created latent_space.html")
        
    except Exception as e:
        print(f"Visualization generation failed: {e}")
