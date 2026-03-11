import requests
import json
import time

url = 'https://graphql.anilist.co'

# Query 1: Find active users
user_search_query = '''
query ($page: Int) {
  Page(page: $page, perPage: 50) {
    pageInfo { hasNextPage }
    users(sort: ID_DESC) { 
      id
      statistics {
        anime {
          count
          statuses { status count }
        }
      }
    }
  }
}
'''

# Query 2: Get user list
list_query = '''
query ($userId: Int) {
  MediaListCollection(userId: $userId, type: ANIME, status: COMPLETED) {
    lists {
      entries {
        userId
        mediaId
        score(format: POINT_10)
        updatedAt
        media {
          title { romaji }
          genres
          meanScore
          popularity
          tags { name rank }
        }
      }
    }
  }
}
'''

def get_clean_temporal_data(target_user_count=50, min_completed=80, max_zeros=10):
    headers = {'Content-Type': 'application/json', 'Accept': 'application/json'}
    final_data = []
    users_saved = 0
    page = 1

    print(f"Starting extraction: Target {target_user_count} users with valid timelines...")

    while users_saved < target_user_count:
        res = requests.post(url, json={'query': user_search_query, 'variables': {'page': page}}, headers=headers)
        if res.status_code != 200: 
            print("API Error, pausing...")
            time.sleep(20)
            continue
        
        users = res.json()['data']['Page']['users']
        
        for user in users:
            # Check basic count first
            comp_stats = [s for s in user['statistics']['anime']['statuses'] if s['status'] == 'COMPLETED']
            count = comp_stats[0]['count'] if comp_stats else 0
            
            if count >= min_completed:
                # Fetch the list to check for updatedAt zeros
                list_res = requests.post(url, json={'query': list_query, 'variables': {'userId': user['id']}}, headers=headers)
                if list_res.status_code == 200:
                    entries = list_res.json()['data']['MediaListCollection']['lists'][0]['entries']
                    
                    # Temporal Filtering Logic
                    zero_count = sum(1 for e in entries if e['updatedAt'] == 0)
                    
                    if zero_count <= max_zeros:
                        final_data.extend(entries)
                        users_saved += 1
                        print(f"User {user['id']} PASSED ({zero_count} zeros). Total Users: {users_saved}/{target_user_count}")
                    else:
                        print(f"User {user['id']} REJECTED ({zero_count} zeros found).")
                
                time.sleep(1) # Safe rate limit
                
            if users_saved >= target_user_count: break
        
        page += 1
        if page > 1000: break # Safety break

    return final_data

# Execute
# We want 50 users who have 80+ completed entries and fewer than 10 '0' timestamps.
high_quality_json = get_clean_temporal_data(target_user_count=100, min_completed=30, max_zeros=0)

with open('anilist_raw_data.json', 'w') as f:
    json.dump(high_quality_json, f, indent=4)

print(f"Saved {len(high_quality_json)} interactions with clean timestamps.")