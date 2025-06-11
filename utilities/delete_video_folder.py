#!/usr/bin/env python3
"""
Simple video crop folder deletion utility.
Finds and deletes the folder containing crops for a specific video.
"""

import os
import sys
import json
import logging
from pathlib import Path
from collections import Counter
from datetime import datetime
import argparse
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db import (
    get_all_labeled_images, 
    delete_crow_embeddings,
    get_embedding_ids_by_image_paths
)

# Configure logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

class VideoFolderDeletor:
    """Delete video crop folders while preserving labeled images."""
    
    def __init__(self, video_path, dry_run=True, preserve_labeled=True):
        self.video_path = video_path
        self.video_name = Path(video_path).stem
        self.dry_run = dry_run
        self.preserve_labeled = preserve_labeled
        
        # Statistics
        self.stats = {
            'folders_found': 0,
            'total_crops_found': 0,
            'labeled_crops_found': 0,
            'unlabeled_crops_found': 0,
            'crops_to_delete': 0,
            'crops_preserved': 0,
            'embeddings_deleted': 0,
            'files_deleted': 0,
            'folders_deleted': 0,
            'errors': 0
        }
        
        # Data
        self.labeled_images = {}
        self.video_folders = []
        self.all_crops = []
        self.crops_to_delete = []
        self.crops_to_preserve = []
        
    def find_video_folders(self):
        """Find folders that match this video name."""
        print(f"🔍 Looking for folders matching video: {self.video_name}")
        
        # Places to look for video folders
        search_locations = [
            Path("crow_crops/videos"),
            Path("videos"),
            Path("processing/videos"),
            Path("crow_crops"),  # Sometimes videos are directly in crow_crops
        ]
        
        # Generate possible folder names (handling truncation)
        possible_names = []
        video_base = self.video_name
        
        # Add progressively shorter versions (in case of truncation)
        for length in range(len(video_base), 10, -1):  # Down to minimum 10 chars
            truncated = video_base[:length]
            possible_names.append(truncated)
        
        print(f"🎯 Searching for folder names: {possible_names[:5]}... (and {len(possible_names)-5} more)")
        
        for search_location in search_locations:
            if not search_location.exists():
                print(f"⏭️ {search_location} doesn't exist")
                continue
                
            print(f"📁 Searching in {search_location}...")
            
            # Look for folders matching any of our possible names
            for folder in search_location.iterdir():
                if not folder.is_dir():
                    continue
                    
                folder_name = folder.name
                
                # Check if this folder matches any of our possible names
                for possible_name in possible_names:
                    if folder_name == possible_name or folder_name.startswith(possible_name):
                        print(f"✅ Found matching folder: {folder}")
                        self.video_folders.append(folder)
                        self.stats['folders_found'] += 1
                        break
        
        if not self.video_folders:
            print(f"❌ No folders found for video: {self.video_name}")
            print("💡 Try checking the actual folder names manually:")
            for search_location in search_locations:
                if search_location.exists():
                    print(f"   ls {search_location}")
        else:
            print(f"📂 Found {len(self.video_folders)} matching folders")
            for folder in self.video_folders:
                print(f"   📁 {folder}")
    
    def analyze_folder_contents(self):
        """Analyze contents of found folders."""
        if not self.video_folders:
            return
            
        # Get labeled images for quick lookup
        if self.preserve_labeled:
            print("📥 Loading labeled images from database...")
            labeled_images_list = get_all_labeled_images()
            self.labeled_images = {img['image_path']: img for img in labeled_images_list}
            print(f"✅ Loaded {len(self.labeled_images)} labeled images")
        
        # Scan all crops in the video folders
        for folder in self.video_folders:
            print(f"\n📂 Analyzing folder: {folder}")
            
            # Find all image files in this folder
            image_files = list(folder.rglob("*.jpg"))
            print(f"   📸 Found {len(image_files)} image files")
            
            for image_file in tqdm(image_files, desc=f"Analyzing {folder.name}"):
                crop_info = {
                    'file_path': str(image_file),
                    'posix_path': image_file.as_posix(),
                    'filename': image_file.name,
                    'folder': str(folder),
                    'size_bytes': image_file.stat().st_size if image_file.exists() else 0,
                    'modified_time': datetime.fromtimestamp(image_file.stat().st_mtime) if image_file.exists() else None
                }
                
                self.all_crops.append(crop_info)
                
                # Check if labeled
                posix_path = crop_info['posix_path']
                if self.preserve_labeled and posix_path in self.labeled_images:
                    label_info = self.labeled_images[posix_path]
                    crop_info['label'] = label_info['label']
                    crop_info['label_confidence'] = label_info.get('confidence')
                    crop_info['is_training_data'] = label_info.get('is_training_data', False)
                    
                    self.crops_to_preserve.append(crop_info)
                    self.stats['labeled_crops_found'] += 1
                else:
                    self.crops_to_delete.append(crop_info)
                    self.stats['unlabeled_crops_found'] += 1
        
        # Update stats
        self.stats['total_crops_found'] = len(self.all_crops)
        self.stats['crops_to_delete'] = len(self.crops_to_delete)
        self.stats['crops_preserved'] = len(self.crops_to_preserve)
    
    def show_analysis_summary(self):
        """Show what was found."""
        print("\n" + "="*60)
        print("📊 FOLDER ANALYSIS SUMMARY")
        print("="*60)
        print(f"🎯 Video: {self.video_name}")
        print(f"📁 Folders found: {self.stats['folders_found']}")
        print(f"📸 Total crops found: {self.stats['total_crops_found']}")
        print(f"🏷️ Labeled crops: {self.stats['labeled_crops_found']}")
        print(f"❌ Unlabeled crops: {self.stats['unlabeled_crops_found']}")
        print(f"🗑️ Crops to delete: {self.stats['crops_to_delete']}")
        print(f"💾 Crops to preserve: {self.stats['crops_preserved']}")
        
        if self.crops_to_preserve:
            print("\n🏷️ PRESERVED CROPS (Labeled):")
            label_counts = Counter(crop.get('label', 'unknown') for crop in self.crops_to_preserve)
            for label, count in label_counts.most_common():
                print(f"  {label}: {count} images")
        
        if self.video_folders:
            print(f"\n📁 FOLDERS TO PROCESS:")
            for folder in self.video_folders:
                folder_crops = [c for c in self.all_crops if c['folder'] == str(folder)]
                folder_to_delete = [c for c in self.crops_to_delete if c['folder'] == str(folder)]
                folder_to_preserve = [c for c in self.crops_to_preserve if c['folder'] == str(folder)]
                print(f"  📁 {folder}")
                print(f"     🗑️ To delete: {len(folder_to_delete)} files")
                print(f"     💾 To preserve: {len(folder_to_preserve)} files")
    
    def delete_database_embeddings(self):
        """Delete embeddings from database."""
        if not self.crops_to_delete:
            print("No crops to delete - skipping embedding deletion")
            return
            
        print(f"\n🗄️ Deleting database embeddings for {len(self.crops_to_delete)} crops...")
        
        crop_paths = [crop['file_path'] for crop in self.crops_to_delete]
        batch_size = 100
        all_embedding_ids = []
        
        for i in tqdm(range(0, len(crop_paths), batch_size), desc="Finding embeddings"):
            batch = crop_paths[i:i + batch_size]
            embedding_mapping = get_embedding_ids_by_image_paths(batch)
            all_embedding_ids.extend(embedding_mapping.values())
        
        if not all_embedding_ids:
            print("No database embeddings found for these crops")
            return
        
        print(f"Found {len(all_embedding_ids)} embeddings to delete")
        
        if not self.dry_run:
            try:
                deleted_count = delete_crow_embeddings(all_embedding_ids)
                self.stats['embeddings_deleted'] = deleted_count
                print(f"✅ Deleted {deleted_count} embeddings from database")
            except Exception as e:
                print(f"❌ Error deleting embeddings: {e}")
                self.stats['errors'] += 1
        else:
            print(f"🔍 DRY RUN: Would delete {len(all_embedding_ids)} embeddings")
    
    def delete_crops_and_folders(self):
        """Delete crop files and empty folders."""
        if not self.crops_to_delete:
            print("No crop files to delete")
            return
            
        print(f"\n🗑️ Deleting {len(self.crops_to_delete)} crop files...")
        
        deleted_count = 0
        for crop in tqdm(self.crops_to_delete, desc="Deleting files"):
            file_path = Path(crop['file_path'])
            
            try:
                if file_path.exists():
                    if not self.dry_run:
                        file_path.unlink()
                        deleted_count += 1
                    else:
                        deleted_count += 1
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")
                self.stats['errors'] += 1
        
        self.stats['files_deleted'] = deleted_count
        
        if not self.dry_run:
            print(f"✅ Deleted {deleted_count} crop files")
        else:
            print(f"🔍 DRY RUN: Would delete {deleted_count} crop files")
        
        # Clean up empty folders
        self._cleanup_empty_folders()
    
    def _cleanup_empty_folders(self):
        """Remove empty folders after deletion."""
        if self.dry_run:
            print("🔍 DRY RUN: Would clean up empty folders")
            return
            
        print("🧹 Cleaning up empty folders...")
        
        folders_deleted = 0
        for folder in self.video_folders:
            try:
                # Check if folder is empty (no preserved crops)
                remaining_crops = [c for c in self.crops_to_preserve if c['folder'] == str(folder)]
                
                if not remaining_crops and folder.exists():
                    # Check if actually empty
                    if not any(folder.iterdir()):
                        folder.rmdir()
                        folders_deleted += 1
                        print(f"  🗑️ Removed empty folder: {folder}")
                    else:
                        print(f"  📁 Folder not empty, keeping: {folder}")
                elif remaining_crops:
                    print(f"  💾 Folder has {len(remaining_crops)} preserved crops, keeping: {folder}")
                    
            except OSError as e:
                print(f"  ❌ Could not remove folder {folder}: {e}")
        
        self.stats['folders_deleted'] = folders_deleted
        if folders_deleted > 0:
            print(f"✅ Removed {folders_deleted} empty folders")
    
    def create_backup_list(self):
        """Create backup list."""
        if not self.crops_to_delete:
            return
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = Path(f"deleted_video_folder_{self.video_name}_{timestamp}.json")
        
        backup_data = {
            'video_name': self.video_name,
            'video_path': self.video_path,
            'deletion_timestamp': timestamp,
            'dry_run': self.dry_run,
            'preserve_labeled': self.preserve_labeled,
            'video_folders': [str(f) for f in self.video_folders],
            'stats': self.stats,
            'deleted_crops': self.crops_to_delete,
            'preserved_crops': self.crops_to_preserve
        }
        
        try:
            with open(backup_file, 'w') as f:
                json.dump(backup_data, f, indent=2, default=str)
            print(f"📄 Created backup list: {backup_file}")
        except Exception as e:
            print(f"Error creating backup list: {e}")
    
    def execute_deletion(self):
        """Execute the deletion process."""
        print("📁 VIDEO FOLDER DELETION")
        print("=" * 60)
        
        # Find the folders
        self.find_video_folders()
        
        if not self.video_folders:
            return
        
        # Analyze contents
        self.analyze_folder_contents()
        
        # Show summary
        self.show_analysis_summary()
        
        if not self.all_crops:
            print("❌ No crops found in the folders.")
            return
        
        # Create backup
        self.create_backup_list()
        
        # Confirm deletion
        if not self.dry_run and self.crops_to_delete:
            response = input(f"\n⚠️ Delete {len(self.crops_to_delete)} crop files and {len(self.video_folders)} folder(s)? (y/N): ")
            if response.lower() != 'y':
                print("❌ Deletion cancelled")
                return
        
        # Execute deletion
        self.delete_database_embeddings()
        self.delete_crops_and_folders()
        
        # Final summary
        print("\n" + "="*60)
        print("🏁 DELETION COMPLETE")
        print("=" * 60)
        print(f"🎯 Video: {self.video_name}")
        print(f"🔧 Mode: {'DRY RUN' if self.dry_run else 'LIVE DELETION'}")
        print(f"🗑️ Files deleted: {self.stats['files_deleted']}")
        print(f"📁 Folders deleted: {self.stats['folders_deleted']}")
        print(f"🗄️ Embeddings deleted: {self.stats['embeddings_deleted']}")
        print(f"💾 Files preserved: {self.stats['crops_preserved']}")
        print(f"❌ Errors: {self.stats['errors']}")

def main():
    parser = argparse.ArgumentParser(description="Delete video crop folders")
    parser.add_argument('video_path', help='Path to video file or video name')
    parser.add_argument('--execute', action='store_true', help='Execute the deletion')
    parser.add_argument('--no-preserve-labeled', action='store_true', help='Delete labeled images too')
    
    args = parser.parse_args()
    
    dry_run = not args.execute
    preserve_labeled = not args.no_preserve_labeled
    
    deletor = VideoFolderDeletor(
        video_path=args.video_path,
        dry_run=dry_run,
        preserve_labeled=preserve_labeled
    )
    
    try:
        deletor.execute_deletion()
    except KeyboardInterrupt:
        print("\n❌ Deletion cancelled by user")

if __name__ == "__main__":
    main() 