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
'''
import os
import json
import pandas as pd
import torch

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
    Loads raw data, metadata, and interactions from the specified directory.
    
    Args:
        data_dir (str): Path to the directory containing the data files.
        
    Returns:
        tuple: (raw_data, metadata_df, interactions_df)
    """
    raw_data_path = os.path.join(data_dir, 'anilist_raw_data.json')
    metadata_path = os.path.join(data_dir, 'anime_metadata.csv')
    interactions_path = os.path.join(data_dir, 'interactions.csv')

    with open(raw_data_path, 'r') as f:
        raw_data = json.load(f)

    metadata_df = pd.read_csv(metadata_path)
    interactions_df = pd.read_csv(interactions_path)

    return raw_data, metadata_df, interactions_df

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

