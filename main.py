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
        
        # Create media ID mappings to match trainer.py
        all_media_ids = sorted(metadata_df['mediaId'].unique())
        media_id_to_idx = {mid: idx for idx, mid in enumerate(all_media_ids)}
        
        # Replicate edge_index exactly as done in trainer.py
        print("Replicating graph structure from trainer...")
        edge_list = []
        user_items = interactions_df.groupby('userId')['mediaId'].apply(list).values
        for items in user_items:
            if len(items) > 1:
                indexed_items = [media_id_to_idx.get(item) for item in items if item in media_id_to_idx]
                for i in range(len(indexed_items) - 1):
                    edge_list.append([indexed_items[i], indexed_items[i+1]])
        
        if edge_list:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.zeros((2, 1), dtype=torch.long)

        print(f"Graph ready: {num_items} nodes, {edge_index.shape[1]} edges.")

        # Extract (node_features will be loaded from checkpoint automatically)
        embeddings = extract_embeddings(
            model_class=AnimeGNN,
            model_path=weights_path,
            num_items=num_items,
            input_dim=input_dim,
            edge_index=edge_index
        )
        
        # Visualize
        fig = create_latent_space_map(embeddings, metadata_df)
        fig.write_html("latent_space.html")
        print("Success! Created latent_space.html")
        
    except Exception as e:
        print(f"Visualization generation failed: {e}")
