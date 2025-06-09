#!/usr/bin/env python3

from db import get_connection, get_all_labeled_images
import os

def cleanup_orphaned_labels(dry_run=True):
    """Remove database entries for files that no longer exist."""
    
    print("🧹 CLEANING ORPHANED DATABASE LABELS")
    print("=" * 50)
    
    labels = get_all_labeled_images()
    print(f"Total labels in database: {len(labels)}")
    
    existing_files = []
    missing_files = []
    
    for label in labels:
        if os.path.exists(label['image_path']):
            existing_files.append(label)
        else:
            missing_files.append(label)
    
    print(f"Existing files: {len(existing_files)}")
    print(f"Missing files: {len(missing_files)}")
    
    if not missing_files:
        print("✅ No orphaned labels found!")
        return
    
    print(f"\n📋 Missing files breakdown:")
    label_counts = {}
    for label in missing_files:
        label_type = label.get('label', 'unknown')
        label_counts[label_type] = label_counts.get(label_type, 0) + 1
    
    for label_type, count in label_counts.items():
        print(f"  {label_type}: {count}")
    
    if dry_run:
        print(f"\n🔍 DRY RUN - Would remove {len(missing_files)} orphaned entries")
        print("Run with dry_run=False to actually remove them")
        return len(missing_files)
    
    # Actually remove orphaned entries
    print(f"\n🗑️ Removing {len(missing_files)} orphaned database entries...")
    
    conn = get_connection()
    cursor = conn.cursor()
    
    removed_count = 0
    for label in missing_files:
        try:
            cursor.execute('DELETE FROM image_labels WHERE image_path = ?', (label['image_path'],))
            removed_count += 1
        except Exception as e:
            print(f"Failed to remove {label['image_path']}: {e}")
    
    conn.commit()
    conn.close()
    
    print(f"✅ Removed {removed_count} orphaned database entries")
    print(f"📊 Database now has {len(existing_files)} valid entries")
    
    return removed_count

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--execute':
        cleanup_orphaned_labels(dry_run=False)
    else:
        cleanup_orphaned_labels(dry_run=True)
        print("\n💡 To actually remove orphaned entries, run:")
        print("python cleanup_orphaned_labels.py --execute") 