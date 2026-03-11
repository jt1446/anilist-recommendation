RNN SEQUENCE DATA FORMAT
==================================================

The sequences are stored as pipe-separated (|) values.

COLUMNS:
  user_id: AniList user ID
  user_name: Username
  total_entries: Length of sequence
  watch_sequence: Pipe-separated anime IDs in chronological order
  score_sequence: Pipe-separated user scores (0-100)
  date_sequence: Pipe-separated dates (YYYY-MM-DD)
  first_date: First date in sequence
  last_date: Last date in sequence

TO LOAD IN PYTHON:
  import pandas as pd
  df = pd.read_csv('watch_sequences.csv')
  # Convert pipe-separated strings to lists
  df['watch_sequence'] = df['watch_sequence'].str.split('|')
  df['watch_sequence'] = df['watch_sequence'].apply(lambda x: [int(i) for i in x])

STATISTICS:
  Total sequences: 74
  Total entries: 42,734
  Average length: 577.5
  Date generated: 2026-03-11 13:30:21
