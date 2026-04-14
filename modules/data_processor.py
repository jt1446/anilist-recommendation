import pandas as pd
import torch
import numpy as np
import json
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler

def build_feature_matrix(metadata_path):
    """Processes genres/tags into a fixed-width numerical tensor."""
    df = pd.read_csv(metadata_path)
    
    # Process genres (Split by '|')
    df['genres'] = df['genres'].fillna('').apply(lambda x: [g.strip() for g in x.split('|')] if x else [])
    mlb_genres = MultiLabelBinarizer()
    genre_features = mlb_genres.fit_transform(df['genres'])
    
    # Process top_tags (Split by '|')
    df['top_tags'] = df['top_tags'].fillna('').apply(lambda x: [t.strip() for t in x.split('|')] if x else [])
    mlb_tags = MultiLabelBinarizer()
    tag_features = mlb_tags.fit_transform(df['top_tags'])
    
    # Process numerical features with StandardScaler
    numerical_cols = ['popularity', 'average_score']
    # Check for available columns to avoid KeyError
    available_cols = []
    if 'popularity' in df.columns:
        available_cols.append('popularity')
    if 'meanScore' in df.columns:
        df['average_score'] = df['meanScore']
        available_cols.append('average_score')
    elif 'average_score' in df.columns:
        available_cols.append('average_score')
        
    if available_cols:
        scaler = StandardScaler()
        # fillna with mean or 0 before scaling
        for col in available_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        numerical_features = scaler.fit_transform(df[available_cols])
    else:
        numerical_features = np.zeros((len(df), 1))
    
    # Combine all features: genres + tags + numerical
    features = np.hstack([genre_features, tag_features, numerical_features])
    features_tensor = torch.tensor(features, dtype=torch.float32)
    
    # Map mediaId to index for future lookups
    id_col = 'mediaId' if 'mediaId' in df.columns else 'anime_id'
    anime_id_to_idx = {id: idx for idx, id in enumerate(df[id_col])}
    
    return features_tensor, anime_id_to_idx

def create_interaction_graph(interactions_path, anime_id_to_idx):
    """Returns an edge index tensor [2, E] for PyTorch Geometric."""
    df = pd.read_csv(interactions_path)
    
    # Map to the column names we actually have
    user_col = 'userId' if 'userId' in df.columns else 'user_id'
    item_col = 'mediaId' if 'mediaId' in df.columns else 'anime_id'
    
    # Filter to only include interactions where we have anime metadata
    df = df[df[item_col].isin(anime_id_to_idx.keys())]
    
    # Map user_id to unique indices for the graph
    unique_users = df[user_col].unique()
    user_id_to_idx = {id: idx for idx, id in enumerate(unique_users)}
    num_users = len(unique_users)
    
    # Shift anime indices to be after user indices in the bipartite graph
    user_indices = [user_id_to_idx[uid] for uid in df[user_col]]
    anime_indices = [anime_id_to_idx[aid] + num_users for aid in df[item_col]]
    
    # Create bidirectional edges
    edge_index = torch.tensor([
        user_indices + anime_indices,
        anime_indices + user_indices
    ], dtype=torch.long)
    
    return edge_index, user_id_to_idx

def split_user_interactions(interactions_df, user_col='userId', item_col='mediaId', time_col='updatedAt', val_count=1, test_count=1):
    """Splits interactions chronologically into train / val / test frames."""
    df = interactions_df.sort_values([user_col, time_col]).copy()

    train_rows = []
    val_rows = []
    test_rows = []

    for user_id, group in df.groupby(user_col):
        n = len(group)
        test_start = max(0, n - test_count)
        val_start = max(0, n - test_count - val_count)

        train_rows.extend(group.iloc[:val_start].to_dict('records'))
        val_rows.extend(group.iloc[val_start:test_start].to_dict('records'))
        test_rows.extend(group.iloc[test_start:].to_dict('records'))

    train_df = pd.DataFrame(train_rows, columns=df.columns)
    val_df = pd.DataFrame(val_rows, columns=df.columns)
    test_df = pd.DataFrame(test_rows, columns=df.columns)

    return train_df, val_df, test_df


def generate_chronological_sequences(interactions_path):
    """Groups interactions by user and sorts by updatedAt."""
    # Note: watch_sequences.csv already contains sequences.
    # We will read from watch_sequences.csv and format it.
    df = pd.read_csv(interactions_path)
    sequences = {}
    
    for _, row in df.iterrows():
        user_id = str(row['user_id'])
        watch_seq = [int(x) for x in str(row['watch_sequence']).split('|')]
        sequences[user_id] = watch_seq
        
    return sequences

if __name__ == "__main__":
    # Example usage/test
    metadata_path = 'johann_data_engineering/processed_data/for_gnn/anime_metadata.csv'
    interactions_path = 'johann_data_engineering/processed_data/for_gnn/user_anime_interactions.csv'
    sequences_path = 'johann_data_engineering/processed_data/for_rnn/watch_sequences.csv'
    
    print("Building feature matrix...")
    features, anime_map = build_feature_matrix(metadata_path)
    torch.save(features, 'anime_features.pt')
    
    print("Creating interaction graph...")
    edge_index, user_map = create_interaction_graph(interactions_path, anime_map)
    torch.save(edge_index, 'adj_matrix.pt')
    
    print("Generating sequences...")
    sequences = generate_chronological_sequences(sequences_path)
    with open('user_sequences.json', 'w') as f:
        json.dump(sequences, f)
    
    print("Done! Files saved.")
