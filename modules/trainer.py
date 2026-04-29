import os
import json
import csv
import pandas as pd
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from .gnn_model import GNNEncoder
from .seq_model import AttentionRNN
from .data_processor import build_feature_matrix, create_interaction_graph, split_user_interactions
from .utils import SequenceDataset, SequenceHoldoutDataset, hit_rate_at_k, mrr_at_k, ndcg_at_k, get_device

# Bayesian Personalized Ranking loss
def bpr_loss(pos_scores, neg_scores):
    """
    Implements the Bayesian Personalized Ranking loss.
    Args:
        pos_scores: [batch_size, 1] - scores for positive items
        neg_scores: [batch_size, num_neg] - scores for negative items
    """
    # Squeeze pos_scores to [batch_size], then unsqueeze to [batch_size, 1] for proper broadcasting
    pos_scores = pos_scores.squeeze(1)  # [batch_size, 1] -> [batch_size]
    pos_scores_exp = pos_scores.unsqueeze(1).expand_as(neg_scores)  # [batch_size, 1] -> [batch_size, num_neg]
    loss = -torch.mean(torch.log(torch.sigmoid(pos_scores_exp - neg_scores) + 1e-8))
    return loss

def train_step(gnn, rnn, batch_data, optimizer, feature_matrix, edge_index):
    """
    Coordinates the forward and backward pass for gradients.
    """
    gnn.train()
    rnn.train()
    optimizer.zero_grad()
    
    sequences, pos_targets, neg_targets = batch_data
    
    all_embeddings = gnn(feature_matrix, edge_index)
    
    batch_size, seq_len = sequences.shape
    seq_embeddings = all_embeddings[sequences.view(-1)].view(batch_size, seq_len, -1)
    
    logits = rnn(seq_embeddings) 
    
    pos_scores = torch.gather(logits, 1, pos_targets.unsqueeze(1)).squeeze(1)
    neg_scores = torch.gather(logits, 1, neg_targets)
    
    loss = bpr_loss(pos_scores, neg_scores)
    loss.backward()
    optimizer.step()
    
    return loss.item()


def build_user_item_pairs(interactions_df, anime_id_to_idx, user_id_to_idx, user_col='userId', item_col='mediaId'):
    pairs = []
    for _, row in interactions_df.iterrows():
        user_id = row[user_col]
        anime_id = row[item_col]
        if user_id in user_id_to_idx and anime_id in anime_id_to_idx:
            pairs.append((user_id_to_idx[user_id], anime_id_to_idx[anime_id]))
    return pairs


def compute_item_popularity(train_df, anime_id_to_idx, item_col='mediaId'):
    item_counts = {idx: 0 for idx in anime_id_to_idx.values()}
    for item_id in train_df[item_col]:
        if item_id in anime_id_to_idx:
            item_counts[anime_id_to_idx[item_id]] += 1
    ranked_items = sorted(item_counts.keys(), key=lambda idx: (-item_counts[idx], idx))
    return ranked_items


def evaluate_constant_ranker(ranked_items, eval_loader, k=10, device='cpu'):
    num_items = len(ranked_items)
    scores = torch.zeros(num_items, dtype=torch.float32, device=device)
    for rank, item_idx in enumerate(ranked_items):
        scores[item_idx] = float(num_items - rank)

    cumulative_hits = 0.0
    cumulative_mrr = 0.0
    cumulative_ndcg = 0.0
    total_examples = 0

    with torch.no_grad():
        for _, targets in eval_loader:
            targets = targets.to(device, non_blocking=True)
            batch_size = targets.size(0)
            logits = scores.unsqueeze(0).expand(batch_size, -1)

            cumulative_hits += hit_rate_at_k(logits, targets, k) * batch_size
            cumulative_mrr += mrr_at_k(logits, targets, k) * batch_size
            cumulative_ndcg += ndcg_at_k(logits, targets, k) * batch_size
            total_examples += batch_size

    if total_examples == 0:
        return 0.0, 0.0, 0.0
    return cumulative_hits / total_examples, cumulative_mrr / total_examples, cumulative_ndcg / total_examples


def append_metrics_csv(path, row):
    write_header = not os.path.exists(path)
    with open(path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def pretrain_gnn(gnn, user_embeddings, features, edge_index, train_pairs, num_users, num_items, device, epochs=5, batch_size=256, num_neg=5):
    optimizer = optim.Adam(
        list(gnn.parameters()) + list(user_embeddings.parameters()),
        lr=1e-3,
        weight_decay=1e-5
    )
    for epoch in range(1, epochs + 1):
        gnn.train()
        total_loss = 0.0
        random.shuffle(train_pairs)

        for start in range(0, len(train_pairs), batch_size):
            batch = train_pairs[start:start + batch_size]
            if len(batch) == 0:
                continue
            users = torch.tensor([u for u, _ in batch], dtype=torch.long, device=device)
            pos_items = torch.tensor([i for _, i in batch], dtype=torch.long, device=device)
            neg_items = torch.randint(0, num_items, (len(batch), num_neg), dtype=torch.long, device=device)

            optimizer.zero_grad()
            full_features = torch.cat([user_embeddings.weight, features], dim=0)
            all_embeddings = gnn(full_features, edge_index)

            user_vecs = all_embeddings[users]
            pos_vecs = all_embeddings[pos_items + num_users]
            neg_vecs = all_embeddings[neg_items + num_users]

            pos_scores = (user_vecs * pos_vecs).sum(dim=1, keepdim=True)
            neg_scores = torch.einsum('bnh,bh->bn', neg_vecs, user_vecs)

            loss = bpr_loss(pos_scores, neg_scores)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(gnn.parameters()) + list(user_embeddings.parameters()),
                max_norm=1.0
            )
            optimizer.step()
            total_loss += loss.item() * len(batch)

        avg_loss = total_loss / len(train_pairs) if len(train_pairs) > 0 else 0.0
        print(f"Pretrain GNN epoch {epoch}/{epochs} - Avg BPR loss: {avg_loss:.4f}")


def evaluate_ranking(gnn, rnn, eval_loader, feature_matrix, edge_index, num_users, k=10, all_embeddings=None):
    """Evaluates ranking metrics on a held-out dataset."""
    gnn.eval()
    rnn.eval()
    cumulative_hits = 0.0
    cumulative_mrr = 0.0
    cumulative_ndcg = 0.0
    total_examples = 0

    with torch.no_grad():
        if all_embeddings is None:
            all_embeddings = gnn(feature_matrix, edge_index)

        for sequences, targets in eval_loader:
            sequences = sequences.to(all_embeddings.device, non_blocking=True)
            targets = targets.to(all_embeddings.device, non_blocking=True)
            batch_size = sequences.shape[0]
            item_seq_indices = sequences + num_users
            seq_embeddings = all_embeddings[item_seq_indices]
            logits = rnn(seq_embeddings)

            cumulative_hits += hit_rate_at_k(logits, targets, k) * batch_size
            cumulative_mrr += mrr_at_k(logits, targets, k) * batch_size
            cumulative_ndcg += ndcg_at_k(logits, targets, k) * batch_size
            total_examples += batch_size

    if total_examples == 0:
        return 0.0, 0.0, 0.0

    return cumulative_hits / total_examples, cumulative_mrr / total_examples, cumulative_ndcg / total_examples

def main():
    """Main training loop for joint GNN + RNN optimization."""
    print("Starting Training Loop...")
    
    # Constants
    DATA_DIR = 'data'
    METADATA_PATH = os.path.join(DATA_DIR, 'anime_metadata.csv')
    INTERACTIONS_PATH = os.path.join(DATA_DIR, 'interactions.csv')
    WEIGHTS_PATH = 'trained_weights.pth'
    
    # 1. Load and Process Data
    device = get_device()
    print(f"Loading data on device: {device}")
    features, anime_id_to_idx = build_feature_matrix(METADATA_PATH)
    edge_index, user_id_to_idx = create_interaction_graph(INTERACTIONS_PATH, anime_id_to_idx)
    features = features.to(device)
    edge_index = edge_index.to(device)
    
    # Load interactions and split into train / val / test
    interactions_df = pd.read_csv(INTERACTIONS_PATH)
    train_df, val_df, test_df = split_user_interactions(
        interactions_df,
        user_col='userId',
        item_col='mediaId',
        time_col='updatedAt',
        val_count=3,
        test_count=3
    )
    print(f"Split interactions: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    num_users = len(user_id_to_idx)
    num_nodes = features.shape[0] + num_users
    num_items = features.shape[0]
    input_dim = features.shape[1]
    hidden_dim = 128

    # Learnable user features improve the bipartite graph representation.
    user_embeddings = nn.Embedding(num_users, input_dim).to(device)
    nn.init.xavier_uniform_(user_embeddings.weight)

    # 2. Initialize Models
    print(f"Initializing models (Input Dim: {input_dim})...")
    gnn = GNNEncoder(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=hidden_dim, dropout=0.3).to(device)
    # Output dim should be num_items for prediction
    rnn = AttentionRNN(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=num_items, dropout=0.4).to(device)

    criterion = nn.CrossEntropyLoss()

    # Prepare datasets
    train_dataset = SequenceDataset(train_df, anime_id_to_idx, seq_len=15)
    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        pin_memory=(device.type == 'cuda'),
        num_workers=2,
        persistent_workers=True
    )
    val_dataset = SequenceHoldoutDataset(interactions_df, val_df, anime_id_to_idx, seq_len=15)
    val_loader = DataLoader(
        val_dataset,
        batch_size=64,
        shuffle=False,
        pin_memory=(device.type == 'cuda'),
        num_workers=2,
        persistent_workers=True
    )
    test_dataset = SequenceHoldoutDataset(interactions_df, test_df, anime_id_to_idx, seq_len=15)
    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
        pin_memory=(device.type == 'cuda'),
        num_workers=2,
        persistent_workers=True
    )

    # Pretrain the GNN on user-item graph structure and fine-tune it during sequence training.
    train_pairs = build_user_item_pairs(train_df, anime_id_to_idx, user_id_to_idx)
    print(f"Pretraining GNN on {len(train_pairs)} user-item interactions...")
    pretrain_gnn(
        gnn,
        user_embeddings,
        features,
        edge_index,
        train_pairs,
        num_users,
        num_items,
        device,
        epochs=5,
        batch_size=512,
        num_neg=5
    )

    # Use joint optimization so the sequence model can fine-tune item embeddings.
    optimizer = optim.Adam(
        list(rnn.parameters()) + list(gnn.parameters()) + list(user_embeddings.parameters()),
        lr=5e-4,
        weight_decay=1e-5
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    # Prepare popularity baseline values for comparison.
    popularity_ranking = compute_item_popularity(train_df, anime_id_to_idx)
    baseline_val_hit, baseline_val_mrr, baseline_val_ndcg = evaluate_constant_ranker(
        popularity_ranking, val_loader, k=10, device=device
    )
    baseline_test_hit, baseline_test_mrr, baseline_test_ndcg = evaluate_constant_ranker(
        popularity_ranking, test_loader, k=10, device=device
    )
    print(
        f"Popularity baseline | Val Hit@10: {baseline_val_hit:.4f}, MRR@10: {baseline_val_mrr:.4f}, NDCG@10: {baseline_val_ndcg:.4f}"
    )
    print(
        f"Popularity baseline | Test Hit@10: {baseline_test_hit:.4f}, MRR@10: {baseline_test_mrr:.4f}, NDCG@10: {baseline_test_ndcg:.4f}"
    )

    metrics_csv_path = 'training_metrics.csv'
    if os.path.exists(metrics_csv_path):
        os.remove(metrics_csv_path)

    # 3. Training Loop
    epochs = 30
    print(f"Performing {epochs} epochs with seq_len=15, Dropout, and LR Scheduling...")

    best_val_hit_rate = 0.0
    best_metrics = (0.0, 0.0, 0.0)

    #scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda')) updated line to fix warning
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))
    
    for epoch in range(1, epochs + 1):
        gnn.train()
        rnn.train()
        total_loss = 0.0

        for sequences, targets in train_loader:
            sequences = sequences.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad()

            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                full_features = torch.cat([user_embeddings.weight, features], dim=0)
                all_embeddings = gnn(full_features, edge_index)
                item_seq_indices = sequences + num_users
                seq_embeddings = all_embeddings[item_seq_indices]

                logits = rnn(seq_embeddings)
                loss = criterion(logits, targets)

            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                list(rnn.parameters()) + list(gnn.parameters()) + list(user_embeddings.parameters()),
                max_norm=1.0
            )

            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader) if len(train_loader) > 0 else 0.0

        with torch.no_grad():
            full_features = torch.cat([user_embeddings.weight, features], dim=0)
            all_embeddings = gnn(full_features, edge_index)

        val_hit_rate, val_mrr, val_ndcg = evaluate_ranking(
            gnn, rnn, val_loader, full_features, edge_index, len(user_id_to_idx), k=10, all_embeddings=all_embeddings
        )

        append_metrics_csv(metrics_csv_path, {
            'epoch': epoch,
            'loss': avg_loss,
            'val_hit_rate': val_hit_rate,
            'val_mrr': val_mrr,
            'val_ndcg': val_ndcg,
            'lr': optimizer.param_groups[0]['lr'],
        })

        scheduler.step(val_hit_rate)

        if val_hit_rate > best_val_hit_rate:
            best_val_hit_rate = val_hit_rate
            best_metrics = (val_hit_rate, val_mrr, val_ndcg)

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"Epoch {epoch}/{epochs} - Avg Loss: {avg_loss:.4f} - "
                f"Val Hit@10: {val_hit_rate:.4f} - MRR@10: {val_mrr:.4f} - NDCG@10: {val_ndcg:.4f} "
                f"(Best: {best_val_hit_rate:.4f})"
            )

    with torch.no_grad():
        full_features = torch.cat([user_embeddings.weight, features], dim=0)
        all_embeddings = gnn(full_features, edge_index)

    test_hit_rate, test_mrr, test_ndcg = evaluate_ranking(
        gnn, rnn, test_loader, full_features, edge_index, len(user_id_to_idx), k=10, all_embeddings=all_embeddings
    )
    print(
        f"Final test metrics: Hit@10 {test_hit_rate:.4f}, MRR@10 {test_mrr:.4f}, NDCG@10 {test_ndcg:.4f}"
    )

    # Save the weights in the format expected by main.py
    print(f"Saving weights to {WEIGHTS_PATH}...")
    torch.save({
        'gnn_state_dict': gnn.state_dict(),
        'rnn_state_dict': rnn.state_dict(),
        'user_embedding_state_dict': user_embeddings.state_dict(),
        'gnn_config': {'input_dim': input_dim, 'hidden_dim': hidden_dim},
        'anime_id_to_idx': anime_id_to_idx,
        'user_id_to_idx': user_id_to_idx,
        'metrics': {
            'best_val_hit_rate': best_metrics[0],
            'best_val_mrr': best_metrics[1],
            'best_val_ndcg': best_metrics[2],
            'test_hit_rate': test_hit_rate,
            'test_mrr': test_mrr,
            'test_ndcg': test_ndcg,
            'baseline_val_hit_rate': baseline_val_hit,
            'baseline_val_mrr': baseline_val_mrr,
            'baseline_val_ndcg': baseline_val_ndcg,
            'baseline_test_hit_rate': baseline_test_hit,
            'baseline_test_mrr': baseline_test_mrr,
            'baseline_test_ndcg': baseline_test_ndcg,
        }
    }, WEIGHTS_PATH)
    print("Training complete.")

if __name__ == "__main__":
    main()
