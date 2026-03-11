#!/usr/bin/env python3
"""
Prepare AniList data for GNN and RNN models
Fixed version with better error handling for None values
"""

import json
import pandas as pd
from pathlib import Path
from datetime import datetime

# Setup paths
PROJECT_FOLDER = Path("/Users/jts/Desktop/AI Neural Networks/Project")
RAW_DATA_FOLDER = PROJECT_FOLDER / "raw_data/anilist_data_20260311_130200"
PROCESSED_FOLDER = PROJECT_FOLDER / "processed_data"

print("="*60)
print("🔄 PREPARING DATA FOR TEAM MEMBERS")
print("="*60)

# Create processed data subfolders
(PROCESSED_FOLDER / "for_gnn").mkdir(parents=True, exist_ok=True)
(PROCESSED_FOLDER / "for_rnn").mkdir(parents=True, exist_ok=True)

# Load manifest
with open(RAW_DATA_FOLDER / "manifest.json", 'r') as f:
    manifest = json.load(f)

print(f"\n📊 Dataset: {manifest['collection_info']['total_collected']} users")
print(f"📊 Total entries: {manifest['summary']['total_entries_sum']}")

# ============================================
# 1. FOR THE GNN ARCHITECT
# ============================================
print("\n🔗 Preparing data for GNN Architect...")

user_item_pairs = []
anime_metadata = {}
anime_id_set = set()

for user in manifest['collected_users']:
    user_file = RAW_DATA_FOLDER / f"user_{user['id']}_{user['name']}.json"
    
    if user_file.exists():
        with open(user_file, 'r', encoding='utf-8') as f:
            user_data = json.load(f)
        
        user_id = user_data['user_id']
        
        # Extract user-anime interactions
        for lst in user_data['data'].get('lists', []):
            for entry in lst.get('entries', []):
                media = entry.get('media', {})
                anime_id = media.get('id')
                
                if anime_id:
                    # Add interaction
                    user_item_pairs.append({
                        'user_id': user_id,
                        'user_name': user['name'],
                        'anime_id': anime_id,
                        'anime_title': media.get('title', {}).get('english') or media.get('title', {}).get('romaji', 'Unknown'),
                        'score': entry.get('score', 0),
                        'status': entry.get('status', 'UNKNOWN'),
                        'progress': entry.get('progress', 0)
                    })
                    
                    # Collect unique anime metadata
                    if anime_id not in anime_id_set:
                        anime_id_set.add(anime_id)
                        
                        # Safely get genres (handle None)
                        genres = media.get('genres', [])
                        if genres is None:
                            genres = []
                        
                        # Safely get studios
                        studios_data = media.get('studios', {})
                        studios = []
                        if studios_data and studios_data.get('nodes'):
                            studios = [s.get('name', '') for s in studios_data['nodes'] if s.get('name')]
                        
                        anime_metadata[anime_id] = {
                            'anime_id': anime_id,
                            'title': media.get('title', {}).get('english') or media.get('title', {}).get('romaji', 'Unknown'),
                            'format': media.get('format', 'Unknown'),
                            'episodes': media.get('episodes', 0) if media.get('episodes') is not None else 0,
                            'duration': media.get('duration', 0) if media.get('duration') is not None else 0,
                            'status': media.get('status', 'Unknown'),
                            'genres': ', '.join(genres) if genres else '',
                            'average_score': media.get('averageScore', 0) if media.get('averageScore') is not None else 0,
                            'popularity': media.get('popularity', 0) if media.get('popularity') is not None else 0,
                            'studios': ', '.join(studios) if studios else ''
                        }

print(f"  ✅ Found {len(user_item_pairs):,} user-anime interactions")
print(f"  ✅ Found {len(anime_metadata):,} unique anime")

# Save GNN data as CSV only
df_interactions = pd.DataFrame(user_item_pairs)
interactions_file = PROCESSED_FOLDER / 'for_gnn/user_anime_interactions.csv'
df_interactions.to_csv(interactions_file, index=False)
print(f"  ✅ Saved interactions to: {interactions_file}")

df_anime = pd.DataFrame(list(anime_metadata.values()))
anime_file = PROCESSED_FOLDER / 'for_gnn/anime_metadata.csv'
df_anime.to_csv(anime_file, index=False)
print(f"  ✅ Saved anime metadata to: {anime_file}")

# ============================================
# 2. FOR THE RNN MODELER (FIXED VERSION)
# ============================================
print("\n⏱️  Preparing data for RNN Modeler...")

rnn_sequences = []
users_with_sequences = 0
total_entries_for_rnn = 0

for user in manifest['collected_users']:
    user_file = RAW_DATA_FOLDER / f"user_{user['id']}_{user['name']}.json"
    
    if user_file.exists():
        with open(user_file, 'r', encoding='utf-8') as f:
            user_data = json.load(f)
        
        # Collect all entries with timestamps
        user_entries = []
        
        for lst in user_data['data'].get('lists', []):
            for entry in lst.get('entries', []):
                media = entry.get('media', {})
                
                # Get timestamp - handle None values safely
                timestamp = 0
                date_str = "Unknown"
                has_valid_date = False
                
                started = entry.get('startedAt', {})
                if started and isinstance(started, dict):
                    year = started.get('year')
                    month = started.get('month')
                    day = started.get('day')
                    
                    # Only create timestamp if we have at least a year
                    if year and year is not None:
                        # Handle None for month and day
                        month_val = 1 if month is None else month
                        day_val = 1 if day is None else day
                        
                        try:
                            timestamp = int(f"{year}{month_val:02d}{day_val:02d}")
                            date_str = f"{year}-{month_val:02d}-{day_val:02d}"
                            has_valid_date = True
                        except (ValueError, TypeError):
                            # If date conversion fails, try createdAt
                            pass
                
                # Fallback to createdAt if no valid startedAt
                if not has_valid_date and entry.get('createdAt'):
                    created_ts = entry.get('createdAt')
                    if created_ts and created_ts is not None:
                        timestamp = created_ts
                        try:
                            date_str = datetime.fromtimestamp(int(created_ts)).strftime('%Y-%m-%d')
                            has_valid_date = True
                        except (ValueError, TypeError, OSError):
                            pass
                
                # Only include entries with valid dates
                if has_valid_date:
                    # Get score safely
                    score = entry.get('score', 0)
                    if score is None:
                        score = 0
                    
                    # Get anime title safely
                    title = 'Unknown'
                    if media and media.get('title'):
                        title = (media.get('title', {}).get('english') or 
                                media.get('title', {}).get('romaji') or 
                                'Unknown')
                    
                    user_entries.append({
                        'anime_id': media.get('id'),
                        'anime_title': title,
                        'timestamp': timestamp,
                        'date': date_str,
                        'score': score
                    })
        
        # Sort by timestamp
        user_entries.sort(key=lambda x: x['timestamp'])
        
        # Only keep users with enough history (at least 10 entries)
        if len(user_entries) >= 10:
            users_with_sequences += 1
            total_entries_for_rnn += len(user_entries)
            
            # Create sequences as pipe-separated strings
            watch_seq = '|'.join([str(e['anime_id']) for e in user_entries])
            score_seq = '|'.join([str(e['score']) for e in user_entries])
            date_seq = '|'.join([e['date'] for e in user_entries])
            
            rnn_sequences.append({
                'user_id': user['id'],
                'user_name': user['name'],
                'total_entries': len(user_entries),
                'watch_sequence': watch_seq,
                'score_sequence': score_seq,
                'date_sequence': date_seq,
                'first_date': user_entries[0]['date'],
                'last_date': user_entries[-1]['date']
            })
            
            # Print progress every 10 users
            if len(rnn_sequences) % 10 == 0:
                print(f"  ... processed {len(rnn_sequences)} user sequences")

# Save RNN data
if rnn_sequences:
    df_sequences = pd.DataFrame(rnn_sequences)
    sequences_file = PROCESSED_FOLDER / 'for_rnn/watch_sequences.csv'
    df_sequences.to_csv(sequences_file, index=False)
    
    print(f"\n  ✅ Created {len(df_sequences)} user sequences for RNN")
    print(f"  ✅ Total entries in sequences: {total_entries_for_rnn:,}")
    print(f"  ✅ Average sequence length: {total_entries_for_rnn/len(df_sequences):.1f}")
    print(f"  ✅ Saved to: {sequences_file}")
    
    # Show sample of first user's sequence
    if len(df_sequences) > 0:
        sample = df_sequences.iloc[0]
        print(f"\n  📝 Sample sequence (first user):")
        print(f"     User: {sample['user_name']}")
        print(f"     Total entries: {sample['total_entries']}")
        print(f"     Date range: {sample['first_date']} to {sample['last_date']}")
        print(f"     First 5 anime IDs: {sample['watch_sequence'].split('|')[:5]}")
    
    # Create a simple README for RNN data
    with open(PROCESSED_FOLDER / 'for_rnn/README.txt', 'w') as f:
        f.write("RNN SEQUENCE DATA FORMAT\n")
        f.write("="*50 + "\n\n")
        f.write("The sequences are stored as pipe-separated (|) values.\n\n")
        f.write("COLUMNS:\n")
        f.write("  user_id: AniList user ID\n")
        f.write("  user_name: Username\n")
        f.write("  total_entries: Length of sequence\n")
        f.write("  watch_sequence: Pipe-separated anime IDs in chronological order\n")
        f.write("  score_sequence: Pipe-separated user scores (0-100)\n")
        f.write("  date_sequence: Pipe-separated dates (YYYY-MM-DD)\n")
        f.write("  first_date: First date in sequence\n")
        f.write("  last_date: Last date in sequence\n\n")
        f.write("TO LOAD IN PYTHON:\n")
        f.write("  import pandas as pd\n")
        f.write("  df = pd.read_csv('watch_sequences.csv')\n")
        f.write("  # Convert pipe-separated strings to lists\n")
        f.write("  df['watch_sequence'] = df['watch_sequence'].str.split('|')\n")
        f.write("  df['watch_sequence'] = df['watch_sequence'].apply(lambda x: [int(i) for i in x])\n\n")
        f.write(f"STATISTICS:\n")
        f.write(f"  Total sequences: {len(df_sequences)}\n")
        f.write(f"  Total entries: {total_entries_for_rnn:,}\n")
        f.write(f"  Average length: {total_entries_for_rnn/len(df_sequences):.1f}\n")
        f.write(f"  Date generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    print(f"  ✅ RNN README saved to: {PROCESSED_FOLDER / 'for_rnn/README.txt'}")
else:
    print("  ⚠️ No valid sequences found for RNN (need at least 10 entries with dates)")

# ============================================
# 3. CREATE A SIMPLE SUMMARY FILE
# ============================================
print("\n📝 Creating summary file...")

summary = {
    'collection_date': manifest['collection_info']['collection_date'],
    'total_users': manifest['collection_info']['total_collected'],
    'total_entries': manifest['summary']['total_entries_sum'],
    'avg_entries_per_user': manifest['summary']['avg_entries_per_user'],
    'unique_anime': len(anime_metadata),
    'total_interactions': len(user_item_pairs),
    'rnn_sequences': len(rnn_sequences) if rnn_sequences else 0,
    'rnn_total_entries': total_entries_for_rnn if rnn_sequences else 0,
    'files_generated': {
        'gnn_interactions': 'for_gnn/user_anime_interactions.csv',
        'gnn_metadata': 'for_gnn/anime_metadata.csv',
        'rnn_sequences': 'for_rnn/watch_sequences.csv' if rnn_sequences else None
    }
}

# Save summary as JSON
with open(PROCESSED_FOLDER / 'data_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

# Save summary as easy-to-read text
with open(PROCESSED_FOLDER / 'README.txt', 'w') as f:
    f.write("ANILIST DATASET - PROCESSED DATA\n")
    f.write("="*60 + "\n\n")
    f.write(f"Collection Date: {summary['collection_date'][:10]}\n")
    f.write(f"Total Users: {summary['total_users']}\n")
    f.write(f"Total Anime Entries: {summary['total_entries']:,}\n")
    f.write(f"Average Entries per User: {summary['avg_entries_per_user']:.1f}\n")
    f.write(f"Unique Anime: {summary['unique_anime']:,}\n")
    f.write(f"Total User-Anime Interactions: {summary['total_interactions']:,}\n\n")
    
    f.write("RNN DATA:\n")
    f.write(f"  • Users with sequences: {summary['rnn_sequences']}\n")
    f.write(f"  • Total entries in sequences: {summary['rnn_total_entries']:,}\n\n")
    
    f.write("FILES GENERATED:\n")
    f.write("-"*40 + "\n")
    f.write("GNN Data (in /for_gnn/):\n")
    f.write(f"  • {summary['files_generated']['gnn_interactions']}\n")
    f.write(f"  • {summary['files_generated']['gnn_metadata']}\n")
    if summary['files_generated']['rnn_sequences']:
        f.write("\nRNN Data (in /for_rnn/):\n")
        f.write(f"  • {summary['files_generated']['rnn_sequences']}\n")
        f.write(f"  • for_rnn/README.txt (format explanation)\n")

print(f"\n✅ Summary saved to: {PROCESSED_FOLDER / 'data_summary.json'}")
print(f"✅ README saved to: {PROCESSED_FOLDER / 'README.txt'}")

print("\n" + "="*60)
print("✅ DATA PREPARATION COMPLETE!")
print("="*60)
print(f"\n📁 Processed data saved to: {PROCESSED_FOLDER}")
print("\n📊 Quick stats:")
print(f"   • {summary['total_interactions']:,} total interactions")
print(f"   • {summary['unique_anime']:,} unique anime")
print(f"   • {summary['rnn_sequences']} RNN sequences")
print(f"   • {summary['rnn_total_entries']:,} entries in RNN sequences")