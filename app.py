import os
import copy
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

# Constants
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

@st.cache_data
def get_base_map(_item_embeddings, _metadata_df):
    """Caches the heavy UMAP/Plotly computation for the Galaxy Map."""
    return create_latent_space_map(_item_embeddings, _metadata_df)

def get_user_history(user_id, interactions_df, metadata_by_id):
    user_df = interactions_df[interactions_df['userId'] == user_id].sort_values('updatedAt')
    history = []
    for _, row in user_df.iterrows():
        media_id = int(row['mediaId'])
        info = metadata_by_id.get(media_id, {})
        history.append({
            'mediaId': media_id,
            'title': format_title(info.get('title', str(media_id))),
            'genres': info.get('genres', ''),
            'primary_genre': info.get('primary_genre', ''),
            'updatedAt': row['updatedAt'],
        })
    return history

def build_recommendations(rnn, sequence_indices, sequence_item_indices, idx_to_media_id, metadata_by_id, top_k=10):
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
        recs.append({
            'mediaId': media_id,
            'title': format_title(info.get('title', str(media_id))),
            'primary_genre': info.get('primary_genre', ''),
            'genres': info.get('genres', ''),
            'score': float(logits[item_idx].cpu().item()),
        })
        if len(recs) >= top_k:
            break
    return recs, attention[0].cpu().tolist()

def highlight_latent_points(fig, history_indices, recommendation_indices):
    fig_copy = copy.deepcopy(fig)
    if len(fig_copy.data) == 0:
        return fig_copy
    
    x, y = list(fig_copy.data[0].x), list(fig_copy.data[0].y)
    
    valid_hist = [i for i in history_indices if 0 <= i < len(x)]
    valid_recs = [i for i in recommendation_indices if 0 <= i < len(x)]

    if valid_hist:
        fig_copy.add_trace(go.Scatter(
            x=[x[i] for i in valid_hist], y=[y[i] for i in valid_hist],
            mode='markers', marker=dict(size=14, color='white', symbol='circle', line=dict(width=2, color='black')),
            name='User History'
        ))
    if valid_recs:
        fig_copy.add_trace(go.Scatter(
            x=[x[i] for i in valid_recs], y=[y[i] for i in valid_recs],
            mode='markers', marker=dict(size=16, color='red', symbol='star'),
            name='Top Recommendations'
        ))
    return fig_copy

def main():
    st.set_page_config(page_title='AniRecc: The Anime Galaxy', layout='wide', initial_sidebar_state="expanded")
    
    # Custom CSS for a cleaner look
    st.markdown("""<style> .main { background-color: #0e1117; } stMetric { background-color: #161b22; border-radius: 10px; padding: 10px; } </style>""", unsafe_allow_html=True)

    st.title('🌌 AniRecc: Deep Learning Anime Galaxy')
    st.sidebar.header("User Selection")

    if not os.path.exists(CHECKPOINT_PATH):
        st.error(f'Checkpoint missing: `{CHECKPOINT_PATH}`. Please upload model weights.')
        return

    # Load Data
    metadata_df, interactions_df, features, anime_id_to_idx, edge_index, user_id_to_idx, metadata_by_id = load_data()
    device = get_device()
    num_users, num_items = len(user_id_to_idx), features.shape[0]
    input_dim, hidden_dim = features.shape[1], 128

    # Load Model
    gnn, rnn, user_embedding = load_model_components(device, num_items, input_dim, hidden_dim, num_users)

    # Sidebar interactions
    user_ids = sorted(interactions_df['userId'].unique().tolist())
    selected_user = st.sidebar.selectbox('Choose a User ID to simulate:', user_ids)

    if selected_user:
        history = get_user_history(selected_user, interactions_df, metadata_by_id)
        recent_history = pd.DataFrame(history).tail(SEQ_LEN)
        
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("📜 Recent Watch History")
            st.dataframe(recent_history[['title', 'primary_genre']].rename(columns={'title': 'Anime', 'primary_genre': 'Genre'}), use_container_width=True)

        # Inference logic
        seq_indices = [anime_id_to_idx[mid] for mid in recent_history['mediaId'] if mid in anime_id_to_idx]
        if seq_indices:
            with torch.no_grad():
                full_features = torch.cat([user_embedding.weight, features.to(device)], dim=0)
                item_embeddings = gnn(full_features, edge_index.to(device))[num_users:]
            
            sequence_tensor = torch.tensor(seq_indices, dtype=torch.long, device=device).unsqueeze(0)
            sequence_embeddings = item_embeddings[sequence_tensor]

            recs, attention_weights = build_recommendations(
                rnn, sequence_embeddings, sequence_tensor,
                {idx: mid for mid, idx in anime_id_to_idx.items()},
                metadata_by_id, top_k=5
            )

            with col2:
                st.subheader("🚀 Predicted Next Watch")
                rec_cols = st.columns(len(recs))
                for i, rec in enumerate(recs):
                    with rec_cols[i]:
                        st.metric(label=f"Rank {i+1}", value=rec['title'][:15]+"...", delta=f"{rec['score']:.2f} score")
                        st.caption(f"Genre: {rec['primary_genre']}")

            # Attention Chart
            st.markdown("---")
            st.subheader("🧠 Explainability: Why these picks?")
            st.info("The Attention mechanism analyzes the user's history. Higher bars indicate which shows 'triggered' the current recommendation.")
            attn_df = pd.DataFrame({'Anime': recent_history['title'].tolist(), 'Influence': attention_weights[:len(recent_history)]})
            st.bar_chart(attn_df.set_index('Anime'))

            # Galaxy Map
            st.subheader("🌌 The Anime Galaxy (Latent Space Visualization)")
            st.write("Each dot is an anime. Distance represents similarity in themes and watching patterns.")
            base_fig = get_base_map(item_embeddings.cpu(), metadata_df)
            final_fig = highlight_latent_points(base_fig, seq_indices, [anime_id_to_idx[r['mediaId']] for r in recs])
            st.plotly_chart(final_fig, use_container_width=True)

    # Technical Expander for Professor
    with st.expander("🛠 Technical Architecture"):
        st.write("""
        **1. GNN Encoder:** Aggregates user-anime interactions to create 128-dim latent embeddings.
        **2. Attention RNN:** Processes the sequence of embeddings to predict the next temporal interaction.
        **3. BPR Loss:** Optimized using Bayesian Personalized Ranking for high-quality recommendation lists.
        """)

if __name__ == '__main__':
    main()
