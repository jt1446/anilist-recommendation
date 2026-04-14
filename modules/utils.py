import os
import json
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset as TorchDataset

def get_device():
    """Checks for available accelerator device and returns it."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

def load_processed_data(data_dir=None):
    """Loads processed data from CSV files."""
    if data_dir is None:
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_dir = os.path.join(BASE_DIR, 'data')
    
    metadata_path = os.path.join(data_dir, 'anime_metadata.csv')
    interactions_path = os.path.join(data_dir, 'interactions.csv')
    
    metadata_df = pd.read_csv(metadata_path)
    interactions_df = pd.read_csv(interactions_path)
    return metadata_df, interactions_df

def map_ids_to_indices(dataframe, id_column='mediaId'):
    """Maps unique IDs to sequential indices."""
    unique_ids = sorted(dataframe[id_column].unique())
    id_to_idx = {id_: idx for idx, id_ in enumerate(unique_ids)}
    idx_to_id = {idx: id_ for id_, idx in id_to_idx.items()}
    mapped_series = dataframe[id_column].map(id_to_idx)
    return mapped_series, id_to_idx, idx_to_id

class SequenceDataset(TorchDataset):
    def __init__(self, interactions_df, anime_id_to_idx, seq_len=5, user_col='userId', item_col='mediaId', time_col='updatedAt'):
        self.seq_len = seq_len
        self.sequences = []
        self.targets = []

        # Sort interactions by user and timestamp
        df = interactions_df.sort_values([user_col, time_col])

        for user_id, group in df.groupby(user_col):
            anime_indices = [anime_id_to_idx[mid] for mid in group[item_col] if mid in anime_id_to_idx]
            if len(anime_indices) > seq_len:
                for i in range(len(anime_indices) - seq_len):
                    self.sequences.append(anime_indices[i:i+seq_len])
                    self.targets.append(anime_indices[i+seq_len])

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx], dtype=torch.long), torch.tensor(self.targets[idx], dtype=torch.long)


class SequenceHoldoutDataset(TorchDataset):
    def __init__(self, interactions_df, holdout_df, anime_id_to_idx, seq_len=5, user_col='userId', item_col='mediaId', time_col='updatedAt'):
        self.seq_len = seq_len
        self.sequences = []
        self.targets = []
        holdout_by_user = holdout_df.groupby(user_col)[item_col].apply(list).to_dict()

        df = interactions_df.sort_values([user_col, time_col])
        for user_id, group in df.groupby(user_col):
            if user_id not in holdout_by_user:
                continue
            anime_indices = [anime_id_to_idx[mid] for mid in group[item_col] if mid in anime_id_to_idx]
            if len(anime_indices) <= seq_len:
                continue
            # Use the final held-out item as the target and the previous seq_len items as input.
            self.sequences.append(anime_indices[-seq_len-1:-1])
            self.targets.append(anime_indices[-1])

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx], dtype=torch.long), torch.tensor(self.targets[idx], dtype=torch.long)


def hit_rate_at_k(logits, targets, k=10):
    topk = torch.topk(logits, k, dim=1).indices
    hits = (topk == targets.unsqueeze(1)).any(dim=1).float()
    return hits.mean().item() if targets.size(0) > 0 else 0.0


def mrr_at_k(logits, targets, k=10):
    topk = torch.topk(logits, k, dim=1).indices
    positions = torch.arange(1, k + 1, device=logits.device).unsqueeze(0).expand_as(topk)
    matches = topk == targets.unsqueeze(1)
    reciprocal = torch.where(matches, 1.0 / positions.float(), torch.tensor(0.0, device=logits.device))
    mrr = reciprocal.max(dim=1)[0]
    return mrr.mean().item() if targets.size(0) > 0 else 0.0


def ndcg_at_k(logits, targets, k=10):
    topk = torch.topk(logits, k, dim=1).indices
    positions = torch.arange(1, k + 1, device=logits.device).unsqueeze(0).expand_as(topk)
    matches = topk == targets.unsqueeze(1)
    ranks = torch.where(matches, positions.float(), torch.tensor(float('inf'), device=logits.device))
    best_rank, _ = ranks.min(dim=1)
    ndcg = torch.where(best_rank.isinf(), torch.tensor(0.0, device=logits.device), 1.0 / torch.log2(best_rank + 1.0))
    return ndcg.mean().item() if targets.size(0) > 0 else 0.0


def extract_embeddings(model, feature_matrix, edge_index):
    """Generates the latest latent vectors for all nodes."""
    model.eval()
    with torch.no_grad():
        embeddings = model(feature_matrix, edge_index)
    return embeddings

def save_checkpoint(model, optimizer, epoch, path):
    """Saves a model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    torch.save(checkpoint, path)

def load_checkpoint(path, model, optimizer=None):
    """Loads a model checkpoint."""
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint.get('epoch', 0)


def load_recommendation_checkpoint(path, gnn, rnn, user_embedding=None, device=None):
    """Loads a recommendation checkpoint with GNN, RNN, and optional user embeddings."""
    if device is None:
        device = torch.device('cpu')

    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except (RuntimeError, pickle.UnpicklingError):
        # The checkpoint may contain NumPy scalar globals that need to be allowlisted.
        if hasattr(torch.serialization, 'safe_globals'):
            with torch.serialization.safe_globals([np._core.multiarray.scalar]):
                checkpoint = torch.load(path, map_location=device, weights_only=False)
        elif hasattr(torch.serialization, 'add_safe_globals'):
            torch.serialization.add_safe_globals([np._core.multiarray.scalar])
            checkpoint = torch.load(path, map_location=device, weights_only=False)
        else:
            checkpoint = torch.load(path, map_location=device, weights_only=False)

    if 'gnn_state_dict' in checkpoint:
        gnn.load_state_dict(checkpoint['gnn_state_dict'])
    elif 'model_state_dict' in checkpoint:
        gnn.load_state_dict(checkpoint['model_state_dict'])
    else:
        raise KeyError('No GNN state dict found in checkpoint')

    if 'rnn_state_dict' in checkpoint:
        rnn.load_state_dict(checkpoint['rnn_state_dict'])
    elif 'model_state_dict' in checkpoint and 'rnn_state_dict' not in checkpoint:
        # fallback to single model checkpoint if available
        pass
    else:
        raise KeyError('No RNN state dict found in checkpoint')

    if user_embedding is not None and 'user_embedding_state_dict' in checkpoint:
        user_embedding.load_state_dict(checkpoint['user_embedding_state_dict'])

    return checkpoint

