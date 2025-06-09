#!/usr/bin/env python3

from db import get_all_labeled_images
import os
from pathlib import Path

def diagnose_missing_files():
    print("🚨 EMERGENCY RECOVERY ANALYSIS")
    print("=" * 50)
    
    labels = get_all_labeled_images()
    print(f"Total labels in database: {len(labels)}")
    
    existing_files = []
    missing_files = []
    
    for label in labels:
        path = label['image_path']
        if os.path.exists(path):
            existing_files.append(label)
        else:
            missing_files.append(label)
    
    print(f"Existing files: {len(existing_files)}")
    print(f"Missing files: {len(missing_files)}")
    
    # Check what was supposed to be from crow_crops2
    crow_crops2_paths = [l for l in labels if 'crow_crops2' in l['image_path']]
    print(f"Database entries that reference crow_crops2: {len(crow_crops2_paths)}")
    
    if crow_crops2_paths:
        print("\nSample crow_crops2 paths in database:")
        for i, label in enumerate(crow_crops2_paths[:5]):
            exists = "✅" if os.path.exists(label['image_path']) else "❌"
            print(f"  {exists} {label['image_path']}")
    
    # Check if there are any orphaned paths that should have been updated
    old_style_paths = [l for l in missing_files if 'crow_crops2' not in l['image_path']]
    print(f"\nMissing files with old paths (should exist in crow_crops): {len(old_style_paths)}")
    
    if old_style_paths:
        print("Sample missing old-style paths:")
        for i, label in enumerate(old_style_paths[:5]):
            print(f"  ❌ {label['image_path']}")
    
    # Check the backup
    backup_dir = Path("backup_all_labeled_crops")
    if backup_dir.exists():
        backup_files = []
        for root, dirs, files in os.walk(backup_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    backup_files.append(os.path.join(root, file))
        print(f"\nBackup contains {len(backup_files)} image files")
        
        # Check if backup has the structure we expect
        crow_crops_backup = backup_dir / "crow_crops"
        crow_crops2_backup = backup_dir / "crow_crops2"
        
        cc_count = 0
        cc2_count = 0
        
        if crow_crops_backup.exists():
            for root, dirs, files in os.walk(crow_crops_backup):
                cc_count += sum(1 for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png')))
        
        if crow_crops2_backup.exists():
            for root, dirs, files in os.walk(crow_crops2_backup):
                cc2_count += sum(1 for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png')))
        
        print(f"  Backup/crow_crops: {cc_count} files")
        print(f"  Backup/crow_crops2: {cc2_count} files") 
        
        return {
            'total_labels': len(labels),
            'existing_files': len(existing_files),
            'missing_files': len(missing_files),
            'backup_cc': cc_count,
            'backup_cc2': cc2_count,
            'missing_list': missing_files[:10]  # First 10 for debugging
        }

if __name__ == "__main__":
    diagnose_missing_files() 