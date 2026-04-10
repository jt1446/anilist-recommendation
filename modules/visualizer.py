'''
Inputs: anime_embeddings.pt, loss_history.csv, anime_metadata.csv

Outputs: Interactive HTML Graph, Matplotlib plots

Key Functions:

    create_latent_space_map(embeddings, metadata): Uses UMAP to create the 2D "Galaxy" view of anime clusters.

    plot_learning_curves(loss_history): Generates the loss and accuracy charts for the 5-page report.

    Class RecommendationInterface(): The Streamlit logic to display results on the web.
'''
import os
import pandas as pd
import numpy as np
import torch
from .utils import get_device, load_processed_data, extract_embeddings, create_latent_space_map

if __name__ == "__main__":
    from .gnn_model import AnimeGNN
    from .utils import load_processed_data, to_edge_index
    import os

    print("Running Full Integration Test (Option 2)...")
    
    try:
        # 1. Load data
        print("Loading processed data...")
        result = load_processed_data()
        if isinstance(result, tuple) and len(result) == 3:
            _, metadata_df, interactions_df = result
        else:
            metadata_df, interactions_df = result
        
        num_items = len(metadata_df)
        print(f"Loaded {num_items} items.")

        # 2. Prepare inputs
        # For a real test, we'd need the actual node features used during training.
        # Use gnn_config if available to set input_dim
        input_dim = 128
        weights_path = os.path.join(os.getcwd(), 'trained_weights.pth')
        if os.path.exists(weights_path):
            try:
                ckpt = torch.load(weights_path, map_location='cpu', weights_only=False)
                if 'gnn_config' in ckpt:
                    input_dim = ckpt['gnn_config'].get('input_dim', input_dim)
            except:
                pass

        node_features = torch.randn(num_items, input_dim)
        
        # Convert interactions to edge_index for the GNN
        # Map IDs to indices (0 to num_items-1) to avoid out-of-bounds errors
        unique_ids = sorted(metadata_df['mediaId'].unique())
        id_to_idx = {id_: idx for idx, id_ in enumerate(unique_ids)}
        
        filtered_interactions = interactions_df[interactions_df['mediaId'].isin(id_to_idx)].copy()
        item_indices = filtered_interactions['mediaId'].map(id_to_idx).values
        
        # GNN expects indices between 0 and num_items-1
        # For simplicity in this test, we'll just use item-item edges based on sequential interactions
        edge_list = []
        for i in range(len(item_indices) - 1):
            edge_list.append([item_indices[i], item_indices[i+1]])
        
        if edge_list:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
        
        # 4. Extract and plot
        embeddings = extract_embeddings(
            model_class=AnimeGNN,
            model_path=weights_path,
            num_items=num_items,
            input_dim=input_dim,
            node_features=node_features,
            edge_index=edge_index
        )
        
        fig = create_latent_space_map(embeddings, metadata_df)
        
        # 5. Save or Show
        output_html = "anime_latent_space.html"
        fig.write_html(output_html)
        print(f"Visualization saved to {output_html}")
        
        # If in an interactive environment, show it
        # fig.show()
        
    except Exception as e:
        print(f"Error during visualization test: {e}")
        import traceback
        traceback.print_exc()
