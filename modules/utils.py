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
        return (torch.tensor(self.sequences[idx], dtype=torch.long),
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
        test_loader: DataLoader for test sequences
        x: Node features
        edge_index: Edge index
        device: Compute device
        k: Top-k for hit rate calculation
    
    Returns:
        hit_rate: Percentage of correct predictions in top-k
    """
    gnn.eval()
    seq_model.eval()
    
    hits = 0
    total = 0
    
    with torch.no_grad():
        item_embeddings = gnn(x, edge_index)
        seq_model.embedding.weight.data[1:] = item_embeddings
        
        for batch_seqs, batch_labels in test_loader:
            batch_seqs = batch_seqs.to(device)
            batch_labels = batch_labels.to(device)
            
            logits = seq_model(batch_seqs)  # [batch, num_items]
            
            # Get top-k predictions
            _, top_k_pred = torch.topk(logits, k=min(k, logits.size(1)), dim=1)
            
            # Check if true label is in top-k
            for i in range(len(batch_labels)):
                if batch_labels[i] in top_k_pred[i]:
                    hits += 1
                total += 1
    
    hit_rate = (hits / total * 100) if total > 0 else 0.0
    gnn.train()
    seq_model.train()
    
    return hit_rate
