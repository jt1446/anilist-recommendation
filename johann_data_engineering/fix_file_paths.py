#!/usr/bin/env python3
"""
Fix file paths in manifest.json to point to correct location
"""

import json
from pathlib import Path

# Setup paths
PROJECT_FOLDER = Path("/Users/jts/Desktop/AI Neural Networks/Project")
MANIFEST_FILE = PROJECT_FOLDER / "raw_data/anilist_data_20260311_130200/manifest.json"
RAW_DATA_FOLDER = PROJECT_FOLDER / "raw_data/anilist_data_20260311_130200"

print("="*60)
print("🔧 FIXING MANIFEST FILE PATHS")
print("="*60)

# Load manifest
with open(MANIFEST_FILE, 'r') as f:
    manifest = json.load(f)

print(f"\n📋 Found {len(manifest['collected_users'])} users in manifest")

# Fix paths
fixed_count = 0
for user in manifest['collected_users']:
    old_path = Path(user['file'])
    correct_path = RAW_DATA_FOLDER / old_path.name
    
    if not old_path.exists() and correct_path.exists():
        user['file'] = str(correct_path)
        fixed_count += 1
        print(f"  ✓ Fixed: {user['name']} -> {correct_path.name}")

# Save fixed manifest
fixed_manifest_file = RAW_DATA_FOLDER / "manifest_fixed.json"
with open(fixed_manifest_file, 'w') as f:
    json.dump(manifest, f, indent=2)

print(f"\n✅ Fixed {fixed_count} file paths")
print(f"✅ Saved fixed manifest to: {fixed_manifest_file}")

# Also create a backup of original
import shutil
shutil.copy2(MANIFEST_FILE, RAW_DATA_FOLDER / "manifest_original.json")
print(f"✅ Backed up original to: {RAW_DATA_FOLDER / 'manifest_original.json'}")