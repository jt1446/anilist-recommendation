'''
provides util functions:
- load_processed_data(path)
- get_device()
- map_ids_to_indices(dataframe)
- reverse_lookup(index, mapping)
- to_edge_index(interactions_df)
- pad_sequences(sequences, max_len)
- save_checkpoint(model, optimizer, epoch, path)
- load_checkpoint(path, model, optimizer)
- log_metrics(metrics_dict, epoch, log_path)
'''