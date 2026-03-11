import requests
import time
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

class AniListCollector:
    def __init__(self, min_entries: int = 30, target_users: int = 75):
        self.base_url = "https://graphql.anilist.co"
        self.min_entries = min_entries
        self.target_users = target_users
        self.session = requests.Session()
        self.request_delay = 1.0  # 1 request per second (safe)
        
        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.data_dir = Path(f"anilist_data_{timestamp}")
        self.data_dir.mkdir(exist_ok=True)
        print(f"📁 Saving data to: {self.data_dir}")
        
    def _make_request(self, query: str, variables: dict) -> Optional[dict]:
        """Make API request with rate limiting"""
        time.sleep(self.request_delay)
        
        try:
            response = self.session.post(
                self.base_url,
                json={"query": query, "variables": variables}
            )
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:
                print("⚠️  Rate limited. Waiting 60 seconds...")
                time.sleep(60)
                return self._make_request(query, variables)
            else:
                print(f"❌ Error {response.status_code}: {response.text}")
                return None
                
        except Exception as e:
            print(f"❌ Request failed: {e}")
            return None
    
    def find_candidates(self, max_pages: int = 20) -> List[dict]:
        """Find users with enough anime entries"""
        query = """
        query ($page: Int) {
            Page(page: $page, perPage: 50) {
                pageInfo {
                    hasNextPage
                    currentPage
                }
                users(sort: [ID]) {
                    id
                    name
                    statistics {
                        anime {
                            count
                        }
                    }
                }
            }
        }
        """
        
        candidates = []
        
        print(f"\n🔍 Scanning for users with {self.min_entries}+ anime entries...")
        print("-" * 60)
        
        for page in range(1, max_pages + 1):
            print(f"  Page {page:2d}: ", end="", flush=True)
            
            result = self._make_request(query, {"page": page})
            
            if not result or not result.get("data", {}).get("Page"):
                print("❌ Failed")
                break
                
            users = result["data"]["Page"]["users"]
            page_candidates = 0
            
            for user in users:
                anime_count = user.get("statistics", {}).get("anime", {}).get("count", 0)
                if anime_count >= self.min_entries:
                    candidates.append({
                        "id": user["id"],
                        "name": user["name"],
                        "anime_count": anime_count
                    })
                    page_candidates += 1
            
            print(f"found {page_candidates} candidates (total: {len(candidates)})")
            
            # Stop if we have enough candidates (2x target for verification)
            if len(candidates) >= self.target_users * 2:
                break
            
            # Stop if no more pages
            if not result["data"]["Page"]["pageInfo"]["hasNextPage"]:
                break
        
        print(f"\n📊 Found {len(candidates)} candidates with {self.min_entries}+ entries")
        return candidates
    
    def verify_and_collect_user(self, user: dict) -> Optional[dict]:
        """Verify and collect full anime list for a user"""
        query = """
        query ($userId: Int) {
            MediaListCollection(userId: $userId, type: ANIME) {
                lists {
                    name
                    entries {
                        id
                        status
                        score(format: POINT_100)
                        progress
                        repeat
                        updatedAt
                        createdAt
                        startedAt { year month day }
                        completedAt { year month day }
                        media {
                            id
                            title {
                                romaji
                                english
                                native
                            }
                            format
                            episodes
                            duration
                            status
                            startDate { year month day }
                            endDate { year month day }
                            season
                            seasonYear
                            averageScore
                            meanScore
                            popularity
                            trending
                            genres
                            tags {
                                name
                                rank
                            }
                            studios(isMain: true) {
                                nodes { name }
                            }
                            siteUrl
                        }
                    }
                }
            }
        }
        """
        
        print(f"    📥 Fetching {user['name']} (ID: {user['id']}, {user['anime_count']} entries)...")
        
        result = self._make_request(query, {"userId": user["id"]})
        
        if not result or not result.get("data", {}).get("MediaListCollection"):
            print(f"    ❌ Failed - private or inaccessible")
            return None
        
        # Count actual entries returned
        lists = result["data"]["MediaListCollection"].get("lists", [])
        total_entries = sum(len(lst.get("entries", [])) for lst in lists)
        
        if total_entries < self.min_entries:
            print(f"    ⚠️  Only {total_entries} entries (needs {self.min_entries})")
            return None
        
        # Prepare output data
        output = {
            "user_id": user["id"],
            "user_name": user["name"],
            "total_entries": total_entries,
            "collection_date": datetime.now().isoformat(),
            "data": result["data"]["MediaListCollection"]
        }
        
        # Save to file
        filename = self.data_dir / f"user_{user['id']}_{user['name']}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        print(f"    ✅ Saved! ({total_entries} entries)")
        
        return {
            "id": user["id"],
            "name": user["name"],
            "entries": total_entries,
            "file": str(filename)
        }
    
    def collect_users(self, candidates: List[dict]) -> List[dict]:
        """Collect data for verified users"""
        collected = []
        
        print(f"\n🔐 Verifying and collecting user data...")
        print("-" * 60)
        
        for i, user in enumerate(candidates, 1):
            if len(collected) >= self.target_users:
                print(f"\n✅ Reached target of {self.target_users} users!")
                break
            
            print(f"  [{len(collected)+1}/{self.target_users}] ", end="")
            result = self.verify_and_collect_user(user)
            
            if result:
                collected.append(result)
            
            # Progress update
            if len(collected) > 0 and len(collected) % 10 == 0:
                print(f"  --- Progress: {len(collected)} users collected ---")
        
        return collected
    
    def create_manifest(self, candidates: List[dict], collected: List[dict]):
        """Create summary manifest file"""
        manifest = {
            "collection_info": {
                "min_entries_per_user": self.min_entries,
                "target_users": self.target_users,
                "collection_date": datetime.now().isoformat(),
                "total_candidates_found": len(candidates),
                "total_collected": len(collected),
                "success_rate": f"{(len(collected)/len(candidates)*100):.1f}%" if candidates else "0%"
            },
            "collected_users": collected,
            "summary": {
                "total_entries_sum": sum(u["entries"] for u in collected),
                "avg_entries_per_user": sum(u["entries"] for u in collected) / len(collected) if collected else 0,
                "max_entries": max((u["entries"] for u in collected), default=0),
                "min_entries": min((u["entries"] for u in collected), default=0)
            }
        }
        
        # Save manifest
        manifest_file = self.data_dir / "manifest.json"
        with open(manifest_file, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        # Also save a simple CSV for easy viewing
        csv_file = self.data_dir / "users_summary.csv"
        with open(csv_file, 'w') as f:
            f.write("user_id,user_name,anime_entries,filename\n")
            for u in collected:
                f.write(f"{u['id']},{u['name']},{u['entries']},{u['file']}\n")
        
        print(f"\n📋 Manifest saved: {manifest_file}")
        print(f"📋 CSV summary: {csv_file}")
        
        return manifest
    
    def run(self):
        """Main execution"""
        print("=" * 60)
        print("🎌 ANILIST DATA COLLECTOR")
        print("=" * 60)
        print(f"Target: {self.target_users} users with {self.min_entries}+ entries each")
        print(f"Rate limit: {self.request_delay}s between requests")
        print("=" * 60)
        
        # Step 1: Find candidates
        candidates = self.find_candidates(max_pages=10)
        
        if not candidates:
            print("❌ No candidates found!")
            return
        
        # Step 2: Collect data
        collected = self.collect_users(candidates)
        
        # Step 3: Create manifest
        manifest = self.create_manifest(candidates, collected)
        
        # Step 4: Final summary
        print("\n" + "=" * 60)
        print("✅ COLLECTION COMPLETE")
        print("=" * 60)
        print(f"📊 Users collected: {len(collected)}/{self.target_users}")
        print(f"📁 Data directory: {self.data_dir}")
        print(f"⭐ Average entries per user: {manifest['summary']['avg_entries_per_user']:.0f}")
        print("=" * 60)

# Run the collector
if __name__ == "__main__":
    collector = AniListCollector(
        min_entries=30,     # Your threshold
        target_users=75      # 50-100 range
    )
    collector.run()