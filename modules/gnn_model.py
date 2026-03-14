'''
Write the GNN model as a notebook in google colab, then download as a .py and put it here.
Make sure to use the functions found in utils.py for loading data, mapping ids, etc.

Input:

    Node Features: A PyTorch tensor (from the Data Scientist) representing anime metadata (genres, tags, etc.).

    Edge Index: A coordinate format (COO) tensor representing the userId to mediaId connections from interactions.csv.

Output:

    Latent Embeddings: A high-dimensional tensor (e.g., 128-dim) where every mediaId is represented by a vector that captures its "position" in the anime graph.
'''
