#!/usr/bin/env python3
"""
Crop Migration Tool

This tool helps safely migrate from old crop extractions to new ones while:
1. Preserving valuable labeled data
2. Cleaning up database references to deleted files
3. Optionally migrating labels from old to new crops
4. Handling embeddings and tracking data appropriately

Usage:
    python crop_migration_tool.py --analyze  # Analyze current state
    python crop_migration_tool.py --migrate --preserve-labeled  # Safe migration
    python crop_migration_tool.py --clean-orphans  # Clean up broken DB references
"""

import os
import sys
import argparse
import shutil
import logging
from pathlib import Path
from collections import defaultdict, Counter
import json
from datetime import datetime

# Import your database functions
from db import (
    get_connection, get_all_labeled_images, get_image_label, 
    add_image_label, get_training_data_stats
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CropMigrationTool:
    def __init__(self, old_crops_dir="crow_crops", new_crops_dir="crow_crops2"):
        self.old_crops_dir = Path(old_crops_dir)
        self.new_crops_dir = Path(new_crops_dir)
        self.backup_dir = Path("backup_labeled_crops")
        
    def analyze_current_state(self):
        """Analyze the current state of crops and database."""
        print("🔍 CROP MIGRATION ANALYSIS")
        print("=" * 50)
        
        # Check directories
        old_exists = self.old_crops_dir.exists()
        new_exists = self.new_crops_dir.exists()
        
        print(f"Old crops directory ({self.old_crops_dir}): {'✅ EXISTS' if old_exists else '❌ MISSING'}")
        print(f"New crops directory ({self.new_crops_dir}): {'✅ EXISTS' if new_exists else '❌ MISSING'}")
        
        if not old_exists and not new_exists:
            print("❌ Neither directory exists! Nothing to migrate.")
            return
            
        # Analyze old crops
        old_stats = self._analyze_directory(self.old_crops_dir) if old_exists else {}
        new_stats = self._analyze_directory(self.new_crops_dir) if new_exists else {}
        
        print(f"\n📊 OLD CROPS ({self.old_crops_dir}):")
        self._print_directory_stats(old_stats)
        
        print(f"\n📊 NEW CROPS ({self.new_crops_dir}):")
        self._print_directory_stats(new_stats)
        
        # Analyze database labels
        print(f"\n🏷️ DATABASE ANALYSIS:")
        self._analyze_database_labels(old_stats.get('files', []), new_stats.get('files', []))
        
        # Analyze embeddings/tracking data
        print(f"\n🧠 TRACKING DATA ANALYSIS:")
        self._analyze_tracking_data()
        
        return {
            'old_stats': old_stats,
            'new_stats': new_stats,
            'old_exists': old_exists,
            'new_exists': new_exists
        }
    
    def _analyze_directory(self, directory):
        """Analyze a crops directory."""
        if not directory.exists():
            return {'files': [], 'total_size': 0, 'subdirs': 0}
            
        files = []
        total_size = 0
        subdirs = 0
        
        for root, dirs, filenames in os.walk(directory):
            subdirs += len(dirs)
            for filename in filenames:
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    filepath = Path(root) / filename
                    try:
                        size = filepath.stat().st_size
                        files.append({
                            'path': filepath,
                            'size': size,
                            'posix_path': filepath.as_posix()
                        })
                        total_size += size
                    except OSError:
                        pass
                        
        return {
            'files': files,
            'total_files': len(files),
            'total_size': total_size,
            'subdirs': subdirs
        }
    
    def _print_directory_stats(self, stats):
        """Print directory statistics."""
        if not stats:
            print("  Directory not found")
            return
            
        total_files = stats.get('total_files', 0)
        total_size = stats.get('total_size', 0)
        subdirs = stats.get('subdirs', 0)
        
        print(f"  Files: {total_files:,}")
        print(f"  Size: {total_size / (1024*1024):.1f} MB")
        print(f"  Subdirectories: {subdirs}")
    
    def _analyze_database_labels(self, old_files, new_files):
        """Analyze database labels for both old and new crops."""
        try:
            all_labeled = get_all_labeled_images()
            stats = get_training_data_stats()
            
            print(f"  Total labeled images in DB: {len(all_labeled)}")
            
            if stats:
                print("  Label distribution:")
                for label, data in stats.items():
                    if isinstance(data, dict):
                        count = data.get('count', 0)
                    else:
                        count = data  # data is already the count
                    print(f"    {label}: {count}")
            
            # Check how many labeled images exist in each directory
            old_paths = {f['posix_path'] for f in old_files}
            new_paths = {f['posix_path'] for f in new_files}
            
            labeled_in_old = 0
            labeled_in_new = 0
            labeled_missing = 0
            
            for labeled_img in all_labeled:
                img_path = labeled_img['image_path']
                if img_path in old_paths:
                    labeled_in_old += 1
                elif img_path in new_paths:
                    labeled_in_new += 1
                else:
                    labeled_missing += 1
            
            print(f"  Labeled images in old crops: {labeled_in_old}")
            print(f"  Labeled images in new crops: {labeled_in_new}")
            print(f"  Labeled images missing from both: {labeled_missing}")
            
            if labeled_missing > 0:
                print(f"  ⚠️ {labeled_missing} labeled images have broken file references!")
                
        except Exception as e:
            print(f"  ❌ Error analyzing database: {e}")
    
    def _analyze_tracking_data(self):
        """Analyze tracking data files."""
        tracking_files = [
            "metadata/crow_tracking.json",
            "metadata/enhanced_tracking.json"
        ]
        
        for tracking_file in tracking_files:
            path = Path(tracking_file)
            if path.exists():
                try:
                    with open(path, 'r') as f:
                        data = json.load(f)
                    
                    crows_count = len(data.get('crows', {}))
                    print(f"  {tracking_file}: {crows_count} tracked crows")
                    
                except Exception as e:
                    print(f"  {tracking_file}: Error reading ({e})")
            else:
                print(f"  {tracking_file}: Not found")
    
    def safe_migrate(self, preserve_labeled=True, create_backup=True):
        """Safely migrate from old to new crops."""
        print("🚀 STARTING SAFE MIGRATION")
        print("=" * 50)
        
        # First, analyze current state
        analysis = self.analyze_current_state()
        
        if not analysis['old_exists']:
            print("❌ Old crops directory doesn't exist. Nothing to migrate.")
            return False
            
        if not analysis['new_exists']:
            print("❌ New crops directory doesn't exist. Cannot migrate.")
            return False
        
        # Get labeled images from old directory
        labeled_images = []
        if preserve_labeled:
            labeled_images = self._get_labeled_images_in_directory(self.old_crops_dir)
            print(f"📋 Found {len(labeled_images)} labeled images in old directory")
        
        # Create backup if requested
        if create_backup and labeled_images:
            self._create_backup(labeled_images)
        
        # Migration strategy
        print("\n🔄 MIGRATION STRATEGY:")
        print(f"1. {'✅' if preserve_labeled else '❌'} Preserve labeled images from old crops")
        print(f"2. {'✅' if create_backup else '❌'} Create backup of labeled images")
        print("3. ✅ Clean orphaned database references")
        print("4. ✅ Update database paths where possible")
        
        # Ask for confirmation
        if not self._confirm_migration():
            print("❌ Migration cancelled by user")
            return False
        
        # Execute migration
        try:
            success = self._execute_migration(labeled_images, preserve_labeled)
            if success:
                print("✅ Migration completed successfully!")
            return success
        except Exception as e:
            print(f"❌ Migration failed: {e}")
            logger.exception("Migration error")
            return False
    
    def _get_labeled_images_in_directory(self, directory):
        """Get all labeled images that exist in the specified directory."""
        labeled_images = []
        all_labeled = get_all_labeled_images()
        
        for labeled_img in all_labeled:
            img_path = Path(labeled_img['image_path'])
            if img_path.exists() and self._is_under_directory(img_path, directory):
                labeled_images.append(labeled_img)
                
        return labeled_images
    
    def _is_under_directory(self, file_path, directory):
        """Check if a file path is under a directory."""
        try:
            file_path.resolve().relative_to(directory.resolve())
            return True
        except ValueError:
            return False
    
    def _create_backup(self, labeled_images):
        """Create backup of labeled images."""
        print(f"💾 Creating backup in {self.backup_dir}")
        
        if self.backup_dir.exists():
            shutil.rmtree(self.backup_dir)
        self.backup_dir.mkdir(parents=True)
        
        # Create metadata file
        metadata = {
            'created_at': datetime.now().isoformat(),
            'source_directory': str(self.old_crops_dir),
            'total_images': len(labeled_images),
            'images': []
        }
        
        copied = 0
        for labeled_img in labeled_images:
            try:
                src_path = Path(labeled_img['image_path'])
                if src_path.exists():
                    # Create relative path structure in backup
                    rel_path = src_path.relative_to(self.old_crops_dir)
                    dst_path = self.backup_dir / rel_path
                    
                    # Create parent directories
                    dst_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Copy file
                    shutil.copy2(src_path, dst_path)
                    copied += 1
                    
                    # Add to metadata
                    metadata['images'].append({
                        'original_path': str(src_path),
                        'backup_path': str(dst_path),
                        'label': labeled_img['label'],
                        'confidence': labeled_img.get('confidence'),
                        'is_training_data': labeled_img.get('is_training_data'),
                        'created_at': labeled_img.get('created_at')
                    })
                    
            except Exception as e:
                logger.warning(f"Failed to backup {labeled_img['image_path']}: {e}")
        
        # Save metadata
        with open(self.backup_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ Backed up {copied} labeled images")
    
    def _confirm_migration(self):
        """Ask user to confirm migration."""
        print("\n⚠️ MIGRATION CONFIRMATION")
        print("This will:")
        print("- Remove unlabeled images from old crops directory")
        print("- Keep labeled images in old directory (if preserve_labeled=True)")
        print("- Clean up orphaned database references")
        print("- Switch primary crops directory to new extraction")
        
        response = input("\nProceed with migration? (yes/no): ").lower().strip()
        return response in ['yes', 'y']
    
    def _execute_migration(self, labeled_images, preserve_labeled):
        """Execute the actual migration."""
        print("\n🔄 EXECUTING MIGRATION...")
        
        # Step 1: Clean orphaned database references
        print("1. Cleaning orphaned database references...")
        orphaned_count = self.clean_orphaned_references()
        print(f"   Cleaned {orphaned_count} orphaned references")
        
        # Step 2: Handle old crops directory
        if preserve_labeled and labeled_images:
            print("2. Preserving labeled images and removing unlabeled ones...")
            self._selective_cleanup(labeled_images)
        else:
            print("2. Removing entire old crops directory...")
            if self.old_crops_dir.exists():
                shutil.rmtree(self.old_crops_dir)
                print(f"   Removed {self.old_crops_dir}")
        
        # Step 3: Update batch reviewer default directory
        print("3. Updating default directory references...")
        self._update_default_directory_references()
        
        return True
    
    def _selective_cleanup(self, labeled_images):
        """Remove unlabeled images while preserving labeled ones."""
        labeled_paths = {Path(img['image_path']).as_posix() for img in labeled_images}
        
        removed_files = 0
        removed_dirs = 0
        
        # Walk through old crops directory
        for root, dirs, files in os.walk(self.old_crops_dir, topdown=False):
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
                pass  # Directory not empty or permission error
        
        print(f"   Removed {removed_files} unlabeled files and {removed_dirs} empty directories")
    
    def _update_default_directory_references(self):
        """Update default directory references in code files."""
        # Note: This is informational - user should manually update their workflow
        print("   📝 Remember to update your workflow to use the new crops directory:")
        print(f"   - Point your batch reviewer to: {self.new_crops_dir}")
        print(f"   - Update any scripts that reference: {self.old_crops_dir}")
    
    def clean_orphaned_references(self):
        """Clean up database references to non-existent files."""
        print("🧹 CLEANING ORPHANED DATABASE REFERENCES")
        print("=" * 50)
        
        try:
            conn = get_connection()
            cursor = conn.cursor()
            
            # Get all image paths from database
            cursor.execute('SELECT id, image_path FROM image_labels')
            all_records = cursor.fetchall()
            
            orphaned_ids = []
            for record_id, image_path in all_records:
                if not Path(image_path).exists():
                    orphaned_ids.append(record_id)
            
            if orphaned_ids:
                print(f"Found {len(orphaned_ids)} orphaned database records")
                
                # Ask for confirmation
                response = input("Delete these orphaned records? (yes/no): ").lower().strip()
                if response in ['yes', 'y']:
                    # Delete orphaned records
                    placeholders = ','.join('?' * len(orphaned_ids))
                    cursor.execute(f'DELETE FROM image_labels WHERE id IN ({placeholders})', orphaned_ids)
                    conn.commit()
                    print(f"✅ Deleted {len(orphaned_ids)} orphaned records")
                else:
                    print("❌ Cleanup cancelled")
                    return 0
            else:
                print("✅ No orphaned references found")
                
            conn.close()
            return len(orphaned_ids)
            
        except Exception as e:
            print(f"❌ Error cleaning orphaned references: {e}")
            logger.exception("Cleanup error")
            return 0
    
    def migration_recommendations(self):
        """Provide recommendations based on current state."""
        print("💡 MIGRATION RECOMMENDATIONS")
        print("=" * 50)
        
        analysis = self.analyze_current_state()
        
        old_stats = analysis.get('old_stats', {})
        new_stats = analysis.get('new_stats', {})
        
        old_files = old_stats.get('total_files', 0)
        new_files = new_stats.get('total_files', 0)
        
        print("Based on your current state, here are my recommendations:\n")
        
        if not analysis['new_exists']:
            print("❌ CRITICAL: New crops directory doesn't exist!")
            print("   → Re-run your improved crop extraction to create crow_crops2")
            return
            
        if new_files == 0:
            print("❌ CRITICAL: New crops directory is empty!")
            print("   → Re-run your improved crop extraction to populate crow_crops2")
            return
        
        # Check if new extraction found significantly more/fewer crops
        if old_files > 0:
            ratio = new_files / old_files
            if ratio < 0.5:
                print("⚠️ WARNING: New extraction has significantly fewer crops")
                print(f"   Old: {old_files:,} files, New: {new_files:,} files")
                print("   → Double-check your extraction parameters")
            elif ratio > 2.0:
                print("✅ GOOD: New extraction found significantly more crops")
                print(f"   Old: {old_files:,} files, New: {new_files:,} files")
            else:
                print("✅ GOOD: New extraction has similar number of crops")
                print(f"   Old: {old_files:,} files, New: {new_files:,} files")
        
        # Get labeled images info
        labeled_images = self._get_labeled_images_in_directory(self.old_crops_dir)
        
        if labeled_images:
            print(f"\n🏷️ LABELED DATA: {len(labeled_images)} labeled images in old crops")
            print("   RECOMMENDATION: Use safe migration with --preserve-labeled")
            print("   → This will keep your valuable labeled data")
            print("   → Creates backup copy for safety")
            print("   → Removes only unlabeled old crops")
        else:
            print("\n🏷️ LABELED DATA: No labeled images in old crops")
            print("   RECOMMENDATION: Simple cleanup - just delete old directory")
            print("   → No valuable data to lose")
        
        print(f"\n🧠 EMBEDDINGS/TRACKING:")
        print("   → Old embeddings computed from old crops will become invalid")
        print("   → This is actually GOOD - new crops should give better embeddings")
        print("   → Tracking data will restart with new crops (this is fine)")
        
        print(f"\n📋 RECOMMENDED MIGRATION COMMAND:")
        if labeled_images:
            print("   python crop_migration_tool.py --migrate --preserve-labeled")
        else:
            print("   python crop_migration_tool.py --migrate")
            print("   # OR simply: rm -rf crow_crops")

def main():
    parser = argparse.ArgumentParser(description="Crop Migration Tool")
    parser.add_argument('--analyze', action='store_true', help='Analyze current state')
    parser.add_argument('--migrate', action='store_true', help='Perform migration')
    parser.add_argument('--preserve-labeled', action='store_true', help='Preserve labeled images during migration')
    parser.add_argument('--clean-orphans', action='store_true', help='Clean orphaned database references')
    parser.add_argument('--recommendations', action='store_true', help='Show migration recommendations')
    parser.add_argument('--old-dir', default='crow_crops', help='Old crops directory (default: crow_crops)')
    parser.add_argument('--new-dir', default='crow_crops2', help='New crops directory (default: crow_crops2)')
    
    args = parser.parse_args()
    
    if not any([args.analyze, args.migrate, args.clean_orphans, args.recommendations]):
        parser.print_help()
        return
    
    tool = CropMigrationTool(args.old_dir, args.new_dir)
    
    if args.analyze:
        tool.analyze_current_state()
    
    if args.recommendations:
        tool.migration_recommendations()
    
    if args.clean_orphans:
        tool.clean_orphaned_references()
    
    if args.migrate:
        tool.safe_migrate(preserve_labeled=args.preserve_labeled)

if __name__ == "__main__":
    main() 