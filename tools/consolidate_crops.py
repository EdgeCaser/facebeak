#!/usr/bin/env python3
"""
Crop Consolidation Tool

Consolidates crow_crops and crow_crops2 into a single directory while:
1. Preserving ALL labeled images from both directories
2. Removing only unlabeled images from old directory
3. Merging everything into one unified dataset
4. Updating database paths accordingly

Usage:
    python consolidate_crops.py --analyze
    python consolidate_crops.py --consolidate --target crow_crops
"""

import os
import sys
import argparse
import shutil
import logging
from pathlib import Path
from collections import defaultdict
import json
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db import get_connection, get_all_labeled_images

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CropConsolidator:
    def __init__(self, old_dir="crow_crops", new_dir="crow_crops2"):
        self.old_dir = Path(old_dir)
        self.new_dir = Path(new_dir)
        
    def analyze_consolidation(self):
        """Analyze what consolidation would involve."""
        print("🔍 CROP CONSOLIDATION ANALYSIS")
        print("=" * 50)
        
        # Get all labeled images
        all_labels = get_all_labeled_images()
        
        old_labels = [l for l in all_labels if 'crow_crops2' not in l['image_path'] and 'crow_crops' in l['image_path']]
        new_labels = [l for l in all_labels if 'crow_crops2' in l['image_path']]
        
        print(f"📊 CURRENT STATE:")
        print(f"  Total labeled images: {len(all_labels)}")
        print(f"  Labels in {self.old_dir}: {len(old_labels)}")
        print(f"  Labels in {self.new_dir}: {len(new_labels)}")
        
        # Count total files
        old_files = self._count_image_files(self.old_dir)
        new_files = self._count_image_files(self.new_dir)
        
        print(f"\n📁 FILE COUNTS:")
        print(f"  Total files in {self.old_dir}: {old_files:,}")
        print(f"  Total files in {self.new_dir}: {new_files:,}")
        print(f"  Labeled files in {self.old_dir}: {len(old_labels):,}")
        print(f"  Labeled files in {self.new_dir}: {len(new_labels):,}")
        print(f"  Unlabeled files in {self.old_dir}: {old_files - len(old_labels):,}")
        print(f"  Unlabeled files in {self.new_dir}: {new_files - len(new_labels):,}")
        
        return {
            'old_labels': old_labels,
            'new_labels': new_labels,
            'old_files': old_files,
            'new_files': new_files
        }
    
    def _count_image_files(self, directory):
        """Count image files in directory."""
        if not directory.exists():
            return 0
            
        count = 0
        for root, dirs, files in os.walk(directory):
            count += sum(1 for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png')))
        return count
    
    def consolidate(self, target_dir="crow_crops", create_backup=True):
        """Consolidate both directories into target directory."""
        print("🚀 STARTING CROP CONSOLIDATION")
        print("=" * 50)
        
        # Determine source and target
        if target_dir == "crow_crops":
            target = self.old_dir
            source = self.new_dir
            print(f"📁 Strategy: Move {self.new_dir} → {self.old_dir}, clean unlabeled from {self.old_dir}")
        else:
            target = self.new_dir  
            source = self.old_dir
            print(f"📁 Strategy: Move {self.old_dir} → {self.new_dir}, clean unlabeled from {self.old_dir}")
        
        # Analyze current state
        analysis = self.analyze_consolidation()
        
        # Get labeled files for both directories
        all_labels = get_all_labeled_images()
        source_labels = [l for l in all_labels if str(source) in l['image_path']]
        target_labels = [l for l in all_labels if str(target) in l['image_path']]
        
        print(f"\n🔄 CONSOLIDATION PLAN:")
        print(f"1. ✅ Create backup of all labeled images")
        print(f"2. ✅ Move all {len(source_labels)} labeled images from {source.name} to {target.name}")
        print(f"3. ✅ Move all unlabeled images from {source.name} to {target.name}")
        print(f"4. ✅ Remove unlabeled images from {target.name} (keep only labeled)")
        print(f"5. ✅ Update database paths")
        print(f"6. ✅ Remove empty {source.name} directory")
        
        # Get confirmation
        response = input(f"\nProceed with consolidation into {target.name}? (yes/no): ").lower().strip()
        if response not in ['yes', 'y']:
            print("❌ Consolidation cancelled")
            return False
        
        try:
            # Step 1: Create backup
            if create_backup:
                self._create_backup(all_labels)
            
            # Step 2: Move all images from source to target
            self._move_directory_contents(source, target)
            
            # Step 3: Clean unlabeled images from target
            self._clean_unlabeled_images(target, target_labels + source_labels)
            
            # Step 4: Update database paths
            self._update_database_paths(source, target)
            
            # Step 5: Remove empty source directory
            if source.exists() and not any(source.iterdir()):
                shutil.rmtree(source)
                print(f"✅ Removed empty {source} directory")
            
            print("✅ Consolidation completed successfully!")
            print(f"📁 All crops are now in: {target}")
            
            return True
            
        except Exception as e:
            print(f"❌ Consolidation failed: {e}")
            logger.exception("Consolidation error")
            return False
    
    def _create_backup(self, all_labels):
        """Create backup of all labeled images."""
        backup_dir = Path("backup_all_labeled_crops")
        print(f"💾 Creating backup in {backup_dir}")
        
        if backup_dir.exists():
            shutil.rmtree(backup_dir)
        backup_dir.mkdir(parents=True)
        
        # Create metadata
        metadata = {
            'created_at': datetime.now().isoformat(),
            'total_images': len(all_labels),
            'images': []
        }
        
        copied = 0
        for label_info in all_labels:
            try:
                src_path = Path(label_info['image_path'])
                if src_path.exists():
                    # Maintain directory structure in backup
                    if 'crow_crops2' in str(src_path):
                        rel_path = src_path.relative_to(self.new_dir)
                        dst_path = backup_dir / "crow_crops2" / rel_path
                    else:
                        rel_path = src_path.relative_to(self.old_dir)
                        dst_path = backup_dir / "crow_crops" / rel_path
                    
                    dst_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src_path, dst_path)
                    copied += 1
                    
                    metadata['images'].append({
                        'original_path': str(src_path),
                        'backup_path': str(dst_path),
                        'label': label_info['label']
                    })
                    
            except Exception as e:
                logger.warning(f"Failed to backup {label_info['image_path']}: {e}")
        
        with open(backup_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ Backed up {copied} labeled images")
    
    def _move_directory_contents(self, source, target):
        """Move all contents from source to target directory."""
        print(f"📁 Moving contents from {source} to {target}")
        
        moved_files = 0
        
        for item in source.iterdir():
            if item.is_file() and item.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                # For individual files, move to target root
                dst_path = target / item.name
                if not dst_path.exists():
                    shutil.move(str(item), str(dst_path))
                    moved_files += 1
                else:
                    # Handle naming conflicts
                    base_name = item.stem
                    suffix = item.suffix
                    counter = 1
                    while dst_path.exists():
                        dst_path = target / f"{base_name}_{counter}{suffix}"
                        counter += 1
                    shutil.move(str(item), str(dst_path))
                    moved_files += 1
                    
            elif item.is_dir():
                # For directories, move entire directory structure
                dst_dir = target / item.name
                if not dst_dir.exists():
                    shutil.move(str(item), str(dst_dir))
                else:
                    # Merge directory contents
                    self._merge_directories(item, dst_dir)
                    shutil.rmtree(item)
        
        print(f"✅ Moved {moved_files} files and directory structures")
    
    def _merge_directories(self, src_dir, dst_dir):
        """Merge contents of src_dir into dst_dir."""
        for item in src_dir.rglob('*'):
            if item.is_file():
                rel_path = item.relative_to(src_dir)
                dst_path = dst_dir / rel_path
                dst_path.parent.mkdir(parents=True, exist_ok=True)
                
                if not dst_path.exists():
                    shutil.copy2(item, dst_path)
                else:
                    # Handle naming conflicts
                    base_name = dst_path.stem
                    suffix = dst_path.suffix
                    counter = 1
                    while dst_path.exists():
                        dst_path = dst_path.parent / f"{base_name}_{counter}{suffix}"
                        counter += 1
                    shutil.copy2(item, dst_path)
    
    def _clean_unlabeled_images(self, directory, labeled_images):
        """Remove unlabeled images from directory."""
        print(f"🧹 Cleaning unlabeled images from {directory}")
        
        # Get set of labeled paths (convert to posix for comparison)
        labeled_paths = {Path(img['image_path']).as_posix() for img in labeled_images}
        
        removed_files = 0
        removed_dirs = 0
        
        # Walk through directory and remove unlabeled images
        for root, dirs, files in os.walk(directory, topdown=False):
            for filename in files:
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    filepath = Path(root) / filename
                    if filepath.as_posix() not in labeled_paths:
                        try:
                            filepath.unlink()
                            removed_files += 1
                        except OSError as e:
                            logger.warning(f"Failed to remove {filepath}: {e}")
            
            # Remove empty directories
            try:
                if not os.listdir(root):
                    os.rmdir(root)
                    removed_dirs += 1
            except OSError:
                pass
        
        print(f"✅ Removed {removed_files} unlabeled files and {removed_dirs} empty directories")
    
    def _update_database_paths(self, old_source, new_target):
        """Update database paths from old source to new target."""
        print(f"🗄️ Updating database paths from {old_source} to {new_target}")
        
        try:
            conn = get_connection()
            cursor = conn.cursor()
            
            # Get all paths that reference the old source directory
            cursor.execute('SELECT id, image_path FROM image_labels WHERE image_path LIKE ?', 
                         (f'%{old_source}%',))
            records = cursor.fetchall()
            
            updated = 0
            for record_id, old_path in records:
                # Replace old source path with new target path
                new_path = old_path.replace(str(old_source), str(new_target))
                
                cursor.execute('UPDATE image_labels SET image_path = ? WHERE id = ?', 
                             (new_path, record_id))
                updated += 1
            
            conn.commit()
            conn.close()
            
            print(f"✅ Updated {updated} database paths")
            
        except Exception as e:
            print(f"❌ Error updating database paths: {e}")
            logger.exception("Database update error")

def main():
    parser = argparse.ArgumentParser(description="Crop Consolidation Tool")
    parser.add_argument('--analyze', action='store_true', help='Analyze consolidation impact')
    parser.add_argument('--consolidate', action='store_true', help='Perform consolidation')
    parser.add_argument('--target', choices=['crow_crops', 'crow_crops2'], default='crow_crops', 
                       help='Target directory for consolidation (default: crow_crops)')
    parser.add_argument('--old-dir', default='crow_crops', help='Old crops directory')
    parser.add_argument('--new-dir', default='crow_crops2', help='New crops directory')
    
    args = parser.parse_args()
    
    if not any([args.analyze, args.consolidate]):
        parser.print_help()
        return
    
    consolidator = CropConsolidator(args.old_dir, args.new_dir)
    
    if args.analyze:
        consolidator.analyze_consolidation()
    
    if args.consolidate:
        consolidator.consolidate(target_dir=args.target)

if __name__ == "__main__":
    from db import get_connection
    from pathlib import Path
    import os

    # Gather all image files in the new dataset structure
    search_dirs = [
        Path('dataset/crows/generic'),
        Path('dataset/not_crow'),
        Path('dataset/not_crow/hard_negatives')
    ]
    all_files = []
    for d in search_dirs:
        if d.exists():
            all_files.extend(list(d.rglob('*.jpg')))
            all_files.extend(list(d.rglob('*.jpeg')))
            all_files.extend(list(d.rglob('*.png')))

    # Map filename to new path
    filename_to_newpath = {f.name: f.as_posix() for f in all_files}

    conn = get_connection()
    cursor = conn.cursor()
    updated = 0
    deduped = 0
    removed = 0
    missing = 0

    for filename, new_path in filename_to_newpath.items():
        # Find all DB entries with this filename (regardless of old path)
        cursor.execute("SELECT id, image_path, updated_at FROM image_labels WHERE image_path LIKE ?", (f"%{filename}",))
        rows = cursor.fetchall()
        if not rows:
            missing += 1
            continue
        # If only one, just update its path
        if len(rows) == 1:
            record_id, old_path, _ = rows[0]
            if old_path != new_path:
                cursor.execute('UPDATE image_labels SET image_path = ? WHERE id = ?', (new_path, record_id))
                updated += 1
            continue
        # If multiple, keep the most recent
        rows.sort(key=lambda r: r[2] or '', reverse=True)  # Sort by updated_at desc
        keep_id, keep_old_path, _ = rows[0]
        if keep_old_path != new_path:
            cursor.execute('UPDATE image_labels SET image_path = ? WHERE id = ?', (new_path, keep_id))
            updated += 1
        # Remove the others
        for record_id, old_path, _ in rows[1:]:
            cursor.execute('DELETE FROM image_labels WHERE id = ?', (record_id,))
            removed += 1
        deduped += 1 if len(rows) > 1 else 0

    conn.commit()
    conn.close()
    print(f"✅ Updated {updated} image paths in database to new dataset/ locations.")
    print(f"🗑️ Deduplicated {deduped} sets of duplicate labels (kept most recent).")
    print(f"❌ Removed {removed} duplicate label entries.")
    print(f"❓ {missing} images in dataset/ had no matching label in the database.") 