import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap.umap_ as umap
import plotly.express as px


def create_latent_space_map(
    embeddings,
    metadata,
    method='umap',
    n_neighbors=15,
    min_dist=0.1,
    pca_components=50,
    tsne_perplexity=30,
    tsne_learning_rate=200,
    tsne_early_exaggeration=12,
    tsne_n_iter=1000,
):
    """
    Creates a 2D latent space map for anime embeddings.

    By default, UMAP is used for a clearer clustering layout. TSNE remains available
    as a fallback for comparison.

    Args:
        embeddings: Tensor or ndarray [num_anime, embedding_dim]
        metadata: DataFrame containing anime titles and genres
        method: 'umap' or 'tsne'
        n_neighbors: UMAP neighbor parameter
        min_dist: UMAP minimum distance parameter
        pca_components: number of PCA components before UMAP/TSNE
        tsne_perplexity: TSNE perplexity
        tsne_learning_rate: TSNE learning rate
        tsne_early_exaggeration: TSNE early exaggeration
        tsne_n_iter: TSNE number of iterations
    Returns:
        fig: Plotly figure object
    """
    if torch.is_tensor(embeddings):
        embeddings = embeddings.detach().cpu().numpy()

    if embeddings.ndim != 2:
        raise ValueError('embeddings must be a 2D array or tensor')

    if embeddings.shape[0] == 0:
        raise ValueError('embeddings must contain at least one sample')

    # Preprocess through PCA to denoise and reduce dimensionality before projection.
    n_pca = min(pca_components, embeddings.shape[1])
    pca = PCA(n_components=n_pca, random_state=42)
    embeddings_preprocessed = pca.fit_transform(embeddings)

    if method == 'umap':
        reducer = umap.UMAP(
            n_components=2,
            random_state=42,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric='cosine'
        )
        embeddings_2d = reducer.fit_transform(embeddings_preprocessed)
    elif method == 'tsne':
        tsne = TSNE(
            n_components=2,
            random_state=42,
            init='pca',
            perplexity=tsne_perplexity,
            learning_rate=tsne_learning_rate,
            early_exaggeration=tsne_early_exaggeration,
            n_iter=tsne_n_iter,
        )
        embeddings_2d = tsne.fit_transform(embeddings_preprocessed)
    else:
        raise ValueError("method must be either 'umap' or 'tsne'")

    # Process genres for coloring
    # Data is using '|' as the separator
    all_genres = metadata['genres'][:len(embeddings)].fillna('')
    primary_genre = all_genres.apply(
        lambda x: x.split('|')[0].strip() if '|' in str(x) else str(x).strip()
    )

    df_viz = pd.DataFrame({
        'x': embeddings_2d[:, 0],
        'y': embeddings_2d[:, 1],
        'title': metadata['title'][:len(embeddings)],
        'primary_genre': primary_genre,
        'all_genres': all_genres,
    })

    fig = px.scatter(
        df_viz,
        x='x',
        y='y',
        hover_name='title',
        color='primary_genre',
        hover_data=['all_genres'],
        title='Anime Latent Space "Galaxy" View (Categorized by Primary Genre)',
        opacity=0.8,
    )
    return fig

def plot_learning_curves(loss_history_path):
    """
    Generates the loss and accuracy charts.
    Args:
        loss_history_path: Path to loss_history.csv
    """
    df = pd.read_csv(loss_history_path)
    plt.figure(figsize=(10, 5))
    plt.plot(df['epoch'], df['loss'], label='Training Loss')
    if 'hit_rate' in df.columns:
        plt.plot(df['epoch'], df['hit_rate'], label='Hit Rate @ 10')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('Learning Curves')
    plt.legend()
    plt.grid(True)
    plt.savefig('learning_curves.png')
    print("Learning curves saved to learning_curves.png")

class RecommendationInterface:
    """The Streamlit logic to display results on the web."""
    def __init__(self):
        pass
    
    def run(self):
        import streamlit as st
        st.title("AniList Recommendation System")
        st.write("This is a placeholder for the Streamlit interface.")
        # In a real scenario, you'd add logic to take user ID,
        # get history, run prediction, and show results.

if __name__ == "__main__":
    print("Visualizer module ready.")
