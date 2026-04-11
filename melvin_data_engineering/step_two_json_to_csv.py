'''
turns raw json from step one into two csvs: interactions and metadata. Uses a separator for genres and tags to handle 
the many-to-many relationship in a single table, but this is just a simplification for this project.
'''
import json
import csv

def process_anilist_to_normalized_tables(input_json, interactions_file, metadata_file):
    with open(input_json, 'r') as f:
        data = json.load(f)

    # Use a dictionary for metadata to ensure we only save each unique anime once
    unique_anime = {}
    interaction_list = []

    for entry in data:
        media_id = entry.get('mediaId')
        media_info = entry.get('media', {})

        # 1. Prepare Interaction Data
        interaction_list.append({
            'userId': entry.get('userId'),
            'mediaId': media_id,
            'score': entry.get('score'),
            'updatedAt': entry.get('updatedAt')
        })

        # 2. Prepare Metadata (only if not already added)
        if media_id not in unique_anime:
            genres_list = media_info.get('genres', [])
            primary_genre = genres_list[0] if genres_list else 'Unknown'
            # We still use a separator, but only in the metadata table
            # Alternatively, you could create a THIRD table for a true many-to-many
            unique_anime[media_id] = {
                'mediaId': media_id,
                'title': media_info['title'].get('romaji'),
                'primary_genre': primary_genre,
                'genres': "|".join(genres_list),
                'popularity': media_info.get('popularity'),
                'meanScore': media_info.get('meanScore'),
                # Extracting top 3 tags for concise features
                'top_tags': "|".join([t['name'] for t in media_info.get('tags', [])[:3]])
            }

    # Write Interactions Table
    with open(interactions_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['userId', 'mediaId', 'score', 'updatedAt'])
        writer.writeheader()
        writer.writerows(interaction_list)

    # Write Metadata Table
    with open(metadata_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['mediaId', 'title', 'primary_genre', 'genres', 'popularity', 'meanScore', 'top_tags'])
        writer.writeheader()
        writer.writerows(unique_anime.values())

    print(f"Success! Created {interactions_file} and {metadata_file}")

# Execute
process_anilist_to_normalized_tables('anilist_raw_data.json', 'interactions.csv', 'anime_metadata.csv')