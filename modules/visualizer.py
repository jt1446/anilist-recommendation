'''
Inputs: anime_embeddings.pt, loss_history.csv, anime_metadata.csv

Outputs: Interactive HTML Graph, Matplotlib plots

Key Functions:

    create_latent_space_map(embeddings, metadata): Uses t-SNE to create the 2D/3D "Galaxy" view of anime clusters.

    plot_learning_curves(loss_history): Generates the loss and accuracy charts for the 5-page report.

    Class RecommendationInterface(): The Streamlit logic to display results on the web.
'''