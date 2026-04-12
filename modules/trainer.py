'''
Training loop to optimize both the GNN and sequence models jointly.
Uses functions found in utils.py for loading data, mapping ids, etc.

Outputs: trained_weights.pth, loss_history.csv

Key Functions:
    - bpr_loss(pos_scores, neg_scores): Implements the Bayesian Personalized Ranking loss
    - train_step(gnn, seq_model, batch_data, optimizer): Coordinates forward/backward passes
    - evaluate_hit_rate(gnn, seq_model, test_loader, device, k=10): Calculates Hit@K metric
'''

import os
import json
import csv
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
import sys

from .gnn_model import AnimeGNN
from .utils import (
    load_processed_data, get_device, map_ids_to_indices, pad_sequences, 
    save_checkpoint, log_metrics, SequenceDataset, bpr_loss, train_step, evaluate_hit_rate
)

# Training hyperparameters
EMBEDDING_DIM = 128
RNN_UNITS = 256
BATCH_SIZE = 1024
EPOCHS = 50
LEARNING_RATE = 0.005
WEIGHT_DECAY = 1e-5
MAX_SEQ_LEN = 20
NUM_NEGATIVES = 5


class SeqRNNModel(nn.Module):
    """
    Sequential RNN model for predicting next media item.
    Used jointly with GNN for embeddings.
    """
    def __init__(self, num_items, embedding_dim=128, rnn_units=256):
        super(SeqRNNModel, self).__init__()
        self.embedding = nn.Embedding(num_items + 1, embedding_dim, padding_idx=0)
        self.rnn = nn.LSTM(embedding_dim, rnn_units, batch_first=True)
        self.fc = nn.Linear(rnn_units, num_items)
        
    def forward(self, sequences):
        """
        Args:
            sequences: [batch_size, seq_len] tensor of item indices
        Returns:
            logits: [batch_size, num_items] prediction scores
        """
        # Embed the sequences
        embedded = self.embedding(sequences)  # [batch, seq_len, emb_dim]
        
        # Pass through RNN (take last hidden state)
        _, (h_n, c_n) = self.rnn(embedded)
        rnn_out = h_n[-1]  # [batch, rnn_units]
        
        # Predict next item
        logits = self.fc(rnn_out)  # [batch, num_items]
        return logits


def main():
    """Main training loop for joint GNN + RNN optimization."""
    print("Initializing training...")
    device = get_device()
    
    # Load processed data
    print("Loading data...")
    try:
        result = load_processed_data()
        # Handle both 2-tuple (metadata_df, interactions_df) and 3-tuple (raw_data, metadata_df, interactions_df)
        if isinstance(result, tuple) and len(result) == 3:
            raw_data, metadata_df, interactions_df = result
        else:
            metadata_df, interactions_df = result
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Creating dummy dataset for demonstration...")
        metadata_df = pd.DataFrame({'mediaId': list(range(1, 101))})
        interactions_df = pd.DataFrame({
            'userId': [1]*50 + [2]*50,
            'mediaId': list(range(1, 51)) + list(range(51, 101))
        })
    
    # Create media ID mappings
    all_media_ids = sorted(metadata_df['mediaId'].unique())
    media_id_to_idx = {mid: idx for idx, mid in enumerate(all_media_ids)}
    idx_to_media_id = {idx: mid for mid, idx in media_id_to_idx.items()}
    num_items = len(all_media_ids)
    
    print(f"Total items: {num_items}")
    
    # Prepare GNN inputs
    print("Preparing GNN inputs...")
    
    # Feature Engineering: Use .str.get_dummies(sep='|') to turn genres and tags into features
    feature_parts = []
    if 'genres' in metadata_df.columns:
        print("Extracting genre features...")
        genre_dummies = metadata_df['genres'].str.get_dummies(sep='|')
        feature_parts.append(genre_dummies)
        print(f"Added {genre_dummies.shape[1]} genre features.")
        
    if 'top_tags' in metadata_df.columns:
        print("Extracting tag features...")
        tag_dummies = metadata_df['top_tags'].str.get_dummies(sep='|')
        feature_parts.append(tag_dummies)
        print(f"Added {tag_dummies.shape[1]} tag features.")

    # 3. Numeric Features: Popularity and Mean Score
    numeric_cols = ['popularity', 'meanScore']
    if all(col in metadata_df.columns for col in numeric_cols):
        print("Scaling numeric features (popularity, meanScore)...")
        scaler = StandardScaler()
        # Handle NaNs if any, though the CSV should be clean
        scaled_values = scaler.fit_transform(metadata_df[numeric_cols].fillna(0))
        numeric_features = pd.DataFrame(scaled_values, columns=numeric_cols, index=metadata_df.index)
        feature_parts.append(numeric_features)
        print("Added scaled numeric features.")

    if feature_parts:
        # Concatenate genres, tags, and numeric features
        combined_features = pd.concat(feature_parts, axis=1)
        x = torch.tensor(combined_features.values, dtype=torch.float).to(device)
        input_dim = x.shape[1]
        print(f"Total input features: {input_dim}")
    else:
        # Fallback if no metadata available
        input_dim = min(num_items, 128)
        x = torch.randn(num_items, input_dim).to(device)
        print(f"Falling back to random features with dim {input_dim}")
    
    # Store node_features in the GNN for later persistence
    node_features = x.detach().cpu()
    
    # Create simple edge index (connect each item to top co-watched items)
    edge_list = []
    user_items = interactions_df.groupby('userId')['mediaId'].apply(list).values
    for items in user_items:
        if len(items) > 1:
            indexed_items = [media_id_to_idx.get(item) for item in items if item in media_id_to_idx]
            # Connect consecutive items in watch history
            for i in range(len(indexed_items) - 1):
                edge_list.append([indexed_items[i], indexed_items[i+1]])
    
    if edge_list:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous().to(device)
    else:
        edge_index = torch.zeros((2, 1), dtype=torch.long).to(device)
    
    print(f"Edge index shape: {edge_index.shape}")
    
    # Load or create sequence data
    print("Loading sequence data...")
    sequences_path = 'johann_data_engineering/processed_data/for_rnn/watch_sequences.csv'
    seq_data = []
    
    if os.path.exists(sequences_path):
        try:
            sequences_df = pd.read_csv(sequences_path)
            for _, row in sequences_df.iterrows():
                seq_str = str(row['watch_sequence'])
                ids = [int(x) for x in seq_str.split('|') if x]
                indexed_ids = [media_id_to_idx[id_] for id_ in ids if id_ in media_id_to_idx]
                
                # Create (input_seq, target) pairs
                if len(indexed_ids) > 1:
                    for i in range(1, len(indexed_ids)):
                        input_seq = indexed_ids[:i][-MAX_SEQ_LEN:]
                        target = indexed_ids[i]
                        seq_data.append((input_seq, target))
        except Exception as e:
            print(f"Error loading sequences: {e}")
    
    if not seq_data:
        print("Creating dummy sequence data...")
        for user_id in interactions_df['userId'].unique()[:10]:
            user_items = interactions_df[interactions_df['userId'] == user_id]['mediaId'].tolist()
            indexed_items = [media_id_to_idx.get(x) for x in user_items if x in media_id_to_idx]
            if len(indexed_items) > 1:
                for i in range(1, len(indexed_items)):
                    seq_data.append((indexed_items[:i], indexed_items[i]))
    
    if not seq_data:
        print("Error: No sequence data available. Exiting.")
        return
    
    print(f"Loaded {len(seq_data)} sequence samples")
    
    # Pad sequences
    seqs = [x[0] for x in seq_data]
    labels = [x[1] for x in seq_data]
    padded_seqs = pad_sequences(seqs, max_len=MAX_SEQ_LEN)
    
    # Create data loaders
    dataset = SequenceDataset(padded_seqs, labels)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print("Initializing models...")
    gnn = AnimeGNN(input_dim=input_dim, hidden_dim=128, output_dim=EMBEDDING_DIM).to(device)
    # Store the node features used during training in the model instance
    gnn.node_features = x
    seq_model = SeqRNNModel(num_items=num_items, embedding_dim=EMBEDDING_DIM, 
                            rnn_units=RNN_UNITS).to(device)
    
    # Set up optimizer and storage for metrics
    optimizer = optim.Adam(
        list(gnn.parameters()) + list(seq_model.parameters()),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )
    
    # Adaptive Learning Rate Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    loss_history = []
    hit_rate_history = []
    
    # Training loop
    print("\nStarting training...")
    for epoch in range(EPOCHS):
        gnn.train()
        seq_model.train()
        
        total_loss = 0
        num_batches = 0
        
        for batch_seqs, batch_labels in train_loader:
            batch_seqs = batch_seqs.to(device)
            batch_labels = batch_labels.to(device)
            
            loss = train_step(gnn, seq_model, x, edge_index, batch_seqs, 
                            batch_labels, optimizer, device)
            
            total_loss += loss
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        hit_rate = evaluate_hit_rate(gnn, seq_model, test_loader, x, edge_index, device, k=10)
        
        # Step the scheduler based on Hit@10
        scheduler.step(hit_rate)
        
        loss_history.append(avg_loss)
        hit_rate_history.append(hit_rate)
        
        # Hit rate is already in percentage format from the utility function
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} | Hit@10: {hit_rate:.2f}%")
    
    # Save trained models
    print("\nSaving trained models...")
    torch.save({
        'gnn_state_dict': gnn.state_dict(),
        'seq_model_state_dict': seq_model.state_dict(),
        'gnn_config': {'input_dim': input_dim, 'output_dim': EMBEDDING_DIM},
        'seq_config': {'num_items': num_items, 'embedding_dim': EMBEDDING_DIM},
        'node_features': gnn.node_features.to('cpu'), # Store actual features
        'media_id_mapping': media_id_to_idx
    }, 'trained_weights.pth')
    
    # Save loss history
    with open('loss_history.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'loss', 'hit_rate_at_10'])
        for epoch, (loss, hit_rate) in enumerate(zip(loss_history, hit_rate_history)):
            writer.writerow([epoch + 1, loss, hit_rate])
    
    print("Training complete! Model saved to trained_weights.pth")
    print("Loss history saved to loss_history.csv")


if __name__ == '__main__':
    main()
