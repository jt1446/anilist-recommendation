import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch

from modules.data_processor import build_feature_matrix, create_interaction_graph
from modules.gnn_model import GNNEncoder
from modules.seq_model import AttentionRNN
from modules.utils import (
    get_device,
    load_processed_data,
    load_recommendation_checkpoint,
)
from modules.visualizer import create_latent_space_map

DATA_DIR = 'data'
CHECKPOINT_PATH = 'trained_weights.pth'
SEQ_LEN = 15
TOP_K = 10


def format_title(title):
    return title if isinstance(title, str) else str(title)


@st.cache_data
def load_data():
    metadata_df, interactions_df = load_processed_data(DATA_DIR)
    features, anime_id_to_idx = build_feature_matrix(os.path.join(DATA_DIR, 'anime_metadata.csv'))
    edge_index, user_id_to_idx = create_interaction_graph(
        os.path.join(DATA_DIR, 'interactions.csv'), anime_id_to_idx
    )

    metadata_by_id = {
        int(row['mediaId']): {
            'title': row['title'],
            'genres': row.get('genres', ''),
            'primary_genre': row.get('primary_genre', ''),
        }
        for _, row in metadata_df.iterrows()
    }

    return metadata_df, interactions_df, features, anime_id_to_idx, edge_index, user_id_to_idx, metadata_by_id


@st.cache_resource
def load_model_components(device, num_items, input_dim, hidden_dim, num_users):
    gnn = GNNEncoder(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=hidden_dim, dropout=0.3).to(device)
    rnn = AttentionRNN(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=num_items, dropout=0.4).to(device)
    user_embedding = torch.nn.Embedding(num_users, input_dim).to(device)
    load_recommendation_checkpoint(CHECKPOINT_PATH, gnn, rnn, user_embedding=user_embedding, device=device)
    gnn.eval()
    rnn.eval()
    return gnn, rnn, user_embedding


def get_user_history(user_id, interactions_df, metadata_by_id):
    user_df = interactions_df[interactions_df['userId'] == user_id].sort_values('updatedAt')
    history = []
    for _, row in user_df.iterrows():
        media_id = int(row['mediaId'])
        info = metadata_by_id.get(media_id, {})
        history.append(
            {
                'mediaId': media_id,
                'title': format_title(info.get('title', str(media_id))),
                'genres': info.get('genres', ''),
                'primary_genre': info.get('primary_genre', ''),
                'updatedAt': row['updatedAt'],
            }
        )
    return history


def build_recommendations(rnn, item_embeddings, sequence_indices, sequence_item_indices, idx_to_media_id, metadata_by_id, top_k=10):
    with torch.no_grad():
        logits, attention = rnn(sequence_indices, return_attention=True)
        logits = logits[0]

    sorted_indices = torch.argsort(logits, descending=True).tolist()
    watched = set(sequence_item_indices[0].tolist())

    recs = []
    for item_idx in sorted_indices:
        if item_idx in watched:
            continue
        media_id = idx_to_media_id[item_idx]
        info = metadata_by_id.get(media_id, {})
        recs.append(
            {
                'mediaId': media_id,
                'title': format_title(info.get('title', str(media_id))),
                'primary_genre': info.get('primary_genre', ''),
                'genres': info.get('genres', ''),
                'score': float(logits[item_idx].cpu().item()),
            }
        )
        if len(recs) >= top_k:
            break

    attention_weights = attention[0].cpu().tolist()
    return recs, attention_weights


def highlight_latent_points(fig, history_indices, recommendation_indices):
    if len(fig.data) == 0:
        return fig
    x = list(fig.data[0].x)
    y = list(fig.data[0].y)

    valid_history_indices = [i for i in history_indices if 0 <= i < len(x)]
    valid_recommendation_indices = [i for i in recommendation_indices if 0 <= i < len(x)]

    if valid_history_indices:
        fig.add_trace(
            go.Scatter(
                x=[x[i] for i in valid_history_indices],
                y=[y[i] for i in valid_history_indices],
                mode='markers',
                marker=dict(size=12, color='black', symbol='x'),
                name='History',
                showlegend=True,
            )
        )

    if valid_recommendation_indices:
        fig.add_trace(
            go.Scatter(
                x=[x[i] for i in valid_recommendation_indices],
                y=[y[i] for i in valid_recommendation_indices],
                mode='markers',
                marker=dict(size=12, color='red', symbol='star'),
                name='Recommendations',
                showlegend=True,
            )
        )

    return fig


def main():
    st.set_page_config(page_title='AniList Recommendation Demo', layout='wide')
    st.title('AniList Recommendation Demo')
    st.write(
        'Select a user, review their recent history, and inspect next-item recommendations with a latent space galaxy view.'
    )

    if not os.path.exists(CHECKPOINT_PATH):
        st.error(f'Checkpoint not found at `{CHECKPOINT_PATH}`. Run training first.')
        return

    metadata_df, interactions_df, features, anime_id_to_idx, edge_index, user_id_to_idx, metadata_by_id = load_data()
    device = get_device()
    num_users = len(user_id_to_idx)
    num_items = features.shape[0]
    input_dim = features.shape[1]
    hidden_dim = 128

    gnn, rnn, user_embedding = load_model_components(device, num_items, input_dim, hidden_dim, num_users)

    user_ids = sorted(interactions_df['userId'].unique().tolist())
    selected_user = st.selectbox('Select User ID', user_ids)

    if selected_user is None:
        return

    history = get_user_history(selected_user, interactions_df, metadata_by_id)
    if not history:
        st.warning('No history found for this user.')
        return

    history_df = pd.DataFrame(history)
    recent_history = history_df.tail(SEQ_LEN)
    st.subheader('Recent Watch History')
    st.table(recent_history[['mediaId', 'title', 'primary_genre', 'genres']].rename(columns={'mediaId': 'Media ID'}))

    seq_indices = [anime_id_to_idx[mid] for mid in recent_history['mediaId'] if mid in anime_id_to_idx]
    if len(seq_indices) == 0:
        st.warning('No valid history items available for prediction.')
        return

    item_embeddings = None
    with torch.no_grad():
        full_features = torch.cat([user_embedding.weight, features.to(device)], dim=0)
        all_embeddings = gnn(full_features, edge_index.to(device))
        item_embeddings = all_embeddings[num_users:]

    sequence_tensor = torch.tensor(seq_indices, dtype=torch.long, device=device).unsqueeze(0)
    sequence_embeddings = item_embeddings[sequence_tensor]

    recs, attention_weights = build_recommendations(
        rnn,
        item_embeddings,
        sequence_embeddings,
        sequence_tensor,
        {idx: media_id for media_id, idx in anime_id_to_idx.items()},
        metadata_by_id,
        top_k=TOP_K,
    )

    st.subheader('Top Recommendations')
    st.table(pd.DataFrame(recs))

    st.subheader('Why? Attention weights for history')
    attention_df = pd.DataFrame(
        {
            'title': recent_history['title'].tolist(),
            'attention': attention_weights[: len(recent_history)],
        }
    )
    st.bar_chart(attention_df.set_index('title'))

    st.subheader('The Galaxy Map')
    fig = create_latent_space_map(item_embeddings.cpu(), metadata_df)
    history_item_indices = [anime_id_to_idx[mid] for mid in recent_history['mediaId'] if mid in anime_id_to_idx]
    recommendation_item_indices = [anime_id_to_idx[rec['mediaId']] for rec in recs if rec['mediaId'] in anime_id_to_idx]
    fig = highlight_latent_points(fig, history_item_indices, recommendation_item_indices)
    st.plotly_chart(fig, use_container_width=True)


if __name__ == '__main__':
    main()
