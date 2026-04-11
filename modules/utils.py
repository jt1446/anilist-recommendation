'''
provides util functions:
- load_processed_data(data_dir)
- get_device()
- map_ids_to_indices(dataframe)
- reverse_lookup(index, mapping)
- to_edge_index(interactions_df)
- pad_sequences(sequences, max_len)
- save_checkpoint(model, optimizer, epoch, path)
- load_checkpoint(path, model, optimizer)
- log_metrics(metrics_dict, epoch, log_path)
- SequenceDataset: PyTorch Dataset for watch sequences
- bpr_loss: Bayesian Personalized Ranking loss
- train_step: Joint training step for GNN + RNN
- evaluate_hit_rate: Hit Rate @ K evaluation metric
'''
import os
import json
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset as TorchDataset

def get_device():
    """
    Checks for available accelerator device (CUDA/MPS) and returns it.
    
    Returns:
        torch.device: The best available device ('cuda', 'mps', or 'cpu').
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")
    return device

# Get the absolute path to the directory where utils.py lives (modules/)
# then go up one level to the project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA_DIR = os.path.join(BASE_DIR, 'data')

def load_processed_data(data_dir=DEFAULT_DATA_DIR):
    """
    Loads processed data from CSV files (anime_metadata.csv and interactions.csv).
    Optionally loads raw JSON data if available.
    
    Args:
        data_dir (str): Path to the directory containing the data files.
        
    Returns:
        tuple: (metadata_df, interactions_df) or (raw_data, metadata_df, interactions_df)
               Returns only CSVs if JSON is unavailable, for backward compatibility.
    """
    metadata_path = os.path.join(data_dir, 'anime_metadata.csv')
    interactions_path = os.path.join(data_dir, 'interactions.csv')
    raw_data_path = os.path.join(data_dir, 'anilist_raw_data.json')
    
    # Load the primary CSV files (required)
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"anime_metadata.csv not found at {metadata_path}")
    if not os.path.exists(interactions_path):
        raise FileNotFoundError(f"interactions.csv not found at {interactions_path}")
    
    metadata_df = pd.read_csv(metadata_path)
    interactions_df = pd.read_csv(interactions_path)
    
    # Load raw JSON data if available (optional)
    raw_data = None
    if os.path.exists(raw_data_path):
        try:
            with open(raw_data_path, 'r') as f:
                raw_data = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not load {raw_data_path}: {e}")
    
    # Return based on what was loaded
    if raw_data is not None:
        return raw_data, metadata_df, interactions_df
    else:
        return metadata_df, interactions_df

def map_ids_to_indices(dataframe, id_column='mediaId'):
    """
    Maps unique IDs in a dataframe column to consistent sequential indices (0 to N-1).
    
    Args:
        dataframe (pd.DataFrame): The dataframe containing IDs.
        id_column (str): The name of the column containing IDs to map.
        
    Returns:
        tuple: (pd.Series, dict, dict)
            - Series of new indices
            - Mapping dictionary {id: index}
            - Reverse mapping dictionary {index: id}
    """
    unique_ids = sorted(dataframe[id_column].unique())
    id_to_idx = {id_: idx for idx, id_ in enumerate(unique_ids)}
    idx_to_id = {idx: id_ for id_, idx in id_to_idx.items()}
    
    mapped_series = dataframe[id_column].map(id_to_idx)
    
    return mapped_series, id_to_idx, idx_to_id
def reverse_lookup(index, mapping):
    """
    Returns the original ID for a given index using the provided mapping.
    
    Args:
        index (int): The index to look up.
        mapping (dict): The dictionary mapping indices to IDs.
        
    Returns:
        The original ID associated with the index.
    """
    return mapping.get(index)

def to_edge_index(interactions_df, user_col='userId', item_col='mediaId'):
    """
    Converts interactions DataFrame into an edge index tensor [2, E].
    
    Args:
        interactions_df (pd.DataFrame): DataFrame containing interaction indices.
        user_col (str): The column name for user indices.
        item_col (str): The column name for item indices.
        
    Returns:
        torch.Tensor: Edge index tensor of shape [2, E].
    """
    user_indices = interactions_df[user_col].values
    item_indices = interactions_df[item_col].values
    
    edge_index = torch.stack([
        torch.tensor(user_indices, dtype=torch.long),
        torch.tensor(item_indices, dtype=torch.long)
    ], dim=0)
    
    return edge_index

def pad_sequences(sequences, max_len, padding_value=0):
    """
    Pads or truncates sequences to a fixed maximum length.
    
    Args:
        sequences (list of lists): Raw list of watch histories.
        max_len (int): The target sequence length.
        padding_value (int): The value to use for mapping (default 0).
        
    Returns:
        torch.Tensor: Padded tensor of shape [num_sequences, max_len].
    """
    padded_results = []
    
    for seq in sequences:
        if len(seq) > max_len:
            # Truncate
            padded_results.append(seq[-max_len:])
        else:
            # Pad
            padded_results.append([padding_value] * (max_len - len(seq)) + seq)
            
    return torch.tensor(padded_results, dtype=torch.long)

def save_checkpoint(model, optimizer, epoch, path):
    """
    Saves a model checkpoint.
    
    Args:
        model (torch.nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer to save.
        epoch (int): The current training epoch.
        path (str): The path to save the checkpoint file to.
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path} at epoch {epoch}")

def load_checkpoint(path, model, optimizer=None):
    """
    Loads a model checkpoint.
    
    Args:
        path (str): Path to the checkpoint file.
        model (torch.nn.Module): The model to load weights into.
        optimizer (torch.optim.Optimizer, optional): The optimizer to load weights into.
        
    Returns:
        int: The epoch the checkpoint was saved at.
    """
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint.get('epoch', 0)

def log_metrics(metrics_dict, epoch, log_path='training_log.csv'):
    """
    Appends training metrics to a CSV file.
    
    Args:
        metrics_dict (dict): Dictionary of metrics (e.g., {'loss': 0.5, 'hit_rate': 0.8}).
        epoch (int): The current training epoch.
        log_path (str): Path to the CSV log file.
    """
    row_data = {'epoch': epoch, **metrics_dict}
    df = pd.DataFrame([row_data])
    
    # If file doesn't exist, write it with headers, else append without headers
    if not os.path.exists(log_path):
        df.to_csv(log_path, index=False)
    else:
        df.to_csv(log_path, mode='a', header=False, index=False)


class SequenceDataset(TorchDataset):
    """Dataset for watch sequences."""
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return (self.sequences[idx].clone().detach().to(dtype=torch.long),
                torch.tensor(self.labels[idx], dtype=torch.long))


def bpr_loss(pos_scores, neg_scores):
    """
    Bayesian Personalized Ranking loss.
    
    Args:
        pos_scores: [batch_size] tensor of positive item scores
        neg_scores: [batch_size, num_negatives] tensor of negative item scores
    
    Returns:
        loss: scalar loss value
    """
    # For each positive, compute max over negatives and calculate margin
    pos_expanded = pos_scores.unsqueeze(1)  # [batch, 1]
    
    # BPR loss: -log(sigmoid(pos - neg))
    # Typically use max negative or mean
    neg_max, _ = torch.max(neg_scores, dim=1)  # [batch]
    margin = pos_expanded.squeeze(1) - neg_max
    
    loss = -torch.mean(torch.log(torch.sigmoid(margin)))
    return loss


def train_step(gnn, seq_model, x, edge_index, batch_seqs, batch_labels, 
               optimizer, device):
    """
    Coordinates forward pass through both models and backward pass for gradients.
    
    Args:
        gnn: GNN model
        seq_model: Sequential RNN model
        x: Node features for GNN
        edge_index: Edge index for GNN
        batch_seqs: [batch_size, seq_len] batch of sequences
        batch_labels: [batch_size] batch of target items
        optimizer: Joint optimizer for both models
        device: Compute device
    
    Returns:
        loss: Training loss value
    """
    optimizer.zero_grad()
    
    # Forward through GNN to get item embeddings
    with torch.no_grad():
        item_embeddings = gnn(x, edge_index)  # [num_items, emb_dim]
    
    # Update seq_model embeddings with GNN embeddings
    seq_model.embedding.weight.data[1:] = item_embeddings
    
    # Forward through sequential model
    logits = seq_model(batch_seqs)  # [batch, num_items]
    
    # Standard cross-entropy loss for next item prediction
    criterion = nn.CrossEntropyLoss()
    loss = criterion(logits, batch_labels)
    
    # Backward and optimize
    loss.backward()
    optimizer.step()
    
    return loss.item()


def evaluate_hit_rate(gnn, seq_model, test_loader, x, edge_index, device, k=10):
    """
    Calculates Hit Rate @ K metric.
    
    Args:
        gnn: GNN model
        seq_model: Sequential RNN model
        test_loader: DataLoader for testing
        x: Node features for GNN
        edge_index: Edge index for GNN
        device: Compute device
        k: Top-K value (default 10)
        
    Returns:
        float: Hit rate value as a percentage.
    """
    gnn.eval()
    seq_model.eval()
    
    hits = 0
    total = 0
    
    with torch.no_grad():
        # Get latest item embeddings
        item_embeddings = gnn(x, edge_index)
        seq_model.embedding.weight.data[1:] = item_embeddings
        
        for batch_seqs, batch_labels in test_loader:
            batch_seqs = batch_seqs.to(device)
            batch_labels = batch_labels.to(device)
            
            logits = seq_model(batch_seqs)
            
            # Get top K indices
            _, top_k_indices = torch.topk(logits, k, dim=1)
            
            # Count hits
            hits += (top_k_indices == batch_labels.unsqueeze(1)).any(dim=1).sum().item()
            total += batch_labels.size(0)
            
    return (hits / total) * 100 if total > 0 else 0


def extract_embeddings(model_class, model_path, num_items, input_dim, edge_index, node_features=None):
    """
    Extracts embeddings from a trained AnimeGNN model.
    """
    device = get_device()
    
    # Load weights
    checkpoint = None
    if os.path.exists(model_path):
        # Using weights_only=False to support numpy globals saved in checkpoints
        if torch.cuda.is_available():
            checkpoint = torch.load(model_path, weights_only=False)
        else:
            checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
        
        # Check if config is in checkpoint to handle size mismatches
        if 'gnn_config' in checkpoint:
            input_dim = checkpoint['gnn_config'].get('input_dim', input_dim)
        
        # Use node_features from checkpoint if not provided
        if node_features is None and 'node_features' in checkpoint:
            node_features = checkpoint['node_features']

    # Initialize model
    model = model_class(input_dim=input_dim).to(device)

    if checkpoint is not None:
        # The checkpoint might be the full state_dict or just the model weights
        if 'gnn_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['gnn_state_dict'], strict=False)
        elif 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
    else:
        print(f"Warning: model_path {model_path} not found.")

    if node_features is None:
        raise ValueError("node_features must be provided or available in checkpoint.")

    model.eval()
    with torch.no_grad():
        # Ensure node_features and edge_index are on the same device as the model
        embeddings = model(node_features.to(device), edge_index.to(device))
    
    return embeddings.cpu().numpy()


def create_latent_space_map(embeddings, metadata_df):
    """
    Creates an interactive latent space visualization using UMAP and Plotly.
    """
    import plotly.express as px
    import umap
    
    # Ensure metadata has columns for tooltips
    possible_tooltip_cols = ['title', 'genres', 'averageScore', 'episodes', 'meanScore', 'popularity']
    tooltip_cols = [col for col in possible_tooltip_cols if col in metadata_df.columns]
    
    print(f"Performing UMAP dimensionality reduction on {embeddings.shape}...")
    reducer = umap.UMAP(n_neighbors=50, min_dist=0.1, n_components=2, metric='cosine')
    umap_embeddings = reducer.fit_transform(embeddings)
    
    # Create a DataFrame for plotting
    plot_df = pd.DataFrame(umap_embeddings, columns=['umap_1', 'umap_2'])
    
    # Merge with metadata
    # The order of embeddings must match the order of items in metadata_df
    metadata_subset = metadata_df.reset_index(drop=True)
    plot_df = pd.concat([plot_df, metadata_subset], axis=1)
    
    # Color mapping column - use primary genre and remove full genres list from tooltips
    color_col = None
    if 'primary_genre' in plot_df.columns:
        color_col = 'primary_genre'
        if 'genres' in tooltip_cols:
            tooltip_cols.remove('genres')
        if 'primary_genre' not in tooltip_cols:
            tooltip_cols.append('primary_genre')
    elif 'genres' in plot_df.columns and not plot_df['genres'].isna().all():
        # Fallback if primary_genre not in CSV
        plot_df['primary_genre'] = plot_df['genres'].apply(
            lambda x: x.split('|')[0].strip() if isinstance(x, str) else 
            (x[0] if isinstance(x, list) and len(x) > 0 else 'Unknown')
        )
        color_col = 'primary_genre'
        if 'genres' in tooltip_cols:
            tooltip_cols.remove('genres')
        if 'primary_genre' not in tooltip_cols:
            tooltip_cols.append('primary_genre')
    elif 'averageScore' in plot_df.columns:
        color_col = 'averageScore'
    
    # Create interactive plot
    fig = px.scatter(
        plot_df, 
        x='umap_1', 
        y='umap_2',
        hover_data=tooltip_cols,
        color=color_col,
        title='Anime Latent Space Visualization (UMAP Projection)',
        labels={'umap_1': 'Dimension 1', 'umap_2': 'Dimension 2'},
        template='plotly_dark'
    )
    
    fig.update_traces(marker=dict(size=4, opacity=0.7))
    fig.update_layout(dragmode='pan')
    
    return fig

