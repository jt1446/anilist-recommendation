import requests

ANILIST_GRAPHQL_URL = 'https://graphql.anilist.co'
HEADERS = {
    'Content-Type': 'application/json',
    'Accept': 'application/json'
}

USER_QUERY = '''
query ($name: String) {
  User(name: $name) {
    id
    name
  }
}
'''

USER_HISTORY_QUERY = '''
query ($userId: Int) {
  MediaListCollection(userId: $userId, type: ANIME, status: COMPLETED) {
    lists {
      entries {
        mediaId
        updatedAt
        media {
          title {
            romaji
            english
            native
          }
          genres
          meanScore
          popularity
          tags {
            name
            rank
          }
        }
      }
    }
  }
}
'''


def _graphql_request(query, variables):
    payload = {'query': query, 'variables': variables}

    try:
        response = requests.post(ANILIST_GRAPHQL_URL, json=payload, headers=HEADERS, timeout=15)
    except requests.RequestException as exc:
        raise ValueError(f"AniList API request failed: {exc}") from exc

    if response.status_code != 200:
        raise ValueError(f"AniList API returned HTTP {response.status_code}: {response.text}")

    result = response.json()
    if 'errors' in result:
        message = result['errors'][0].get('message', 'Unknown AniList API error')
        raise ValueError(f"AniList API error: {message}")

    return result.get('data', {})


def resolve_username_to_id(username):
    if not username:
        raise ValueError('AniList username cannot be empty.')

    data = _graphql_request(USER_QUERY, {'name': username})
    user = data.get('User')

    if not user or not user.get('id'):
        raise ValueError(f'AniList user "{username}" was not found.')

    return {'id': int(user['id']), 'name': user.get('name', username)}


def fetch_completed_user_history(user_id):
    data = _graphql_request(USER_HISTORY_QUERY, {'userId': int(user_id)})
    collection = data.get('MediaListCollection')

    if not collection or not collection.get('lists'):
        raise ValueError('Could not load AniList watch history. The profile may be private or empty.')

    entries = []
    for media_list in collection['lists']:
        for entry in media_list.get('entries', []):
            media = entry.get('media')
            if not media:
                continue

            media_id = entry.get('mediaId')
            if media_id is None:
                continue

            title_info = media.get('title') or {}
            title = title_info.get('romaji') or title_info.get('english') or title_info.get('native') or str(media_id)
            genres = media.get('genres') or []
            primary_genre = genres[0] if genres else ''

            entries.append({
                'mediaId': int(media_id),
                'title': title,
                'genres': genres,
                'primary_genre': primary_genre,
                'updatedAt': int(entry.get('updatedAt') or 0),
            })

    if not entries:
        raise ValueError('No completed anime entries were returned by AniList.')

    entries.sort(key=lambda item: item.get('updatedAt', 0) or 0)
    return entries


def fetch_anilist_user_history(username):
    user = resolve_username_to_id(username)
    history = fetch_completed_user_history(user['id'])
    return history, user['name']
