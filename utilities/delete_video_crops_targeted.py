#!/usr/bin/env python3
"""
Targeted video crop deletion utility that handles huge directories smartly.
"""

import os
import sys
import json
import logging
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime
import argparse
import glob
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db import (
    get_connection, 
    get_all_labeled_images, 
    delete_crow_embeddings,
    get_embedding_ids_by_image_paths
)

# Configure logging
logging.basicConfig(level=logging.WARNING)  # Reduce noise
logger = logging.getLogger(__name__)

class TargetedVideoCropDeletor:
    """Targeted deletion that handles huge directories efficiently."""
    
    def __init__(self, video_path, dry_run=True, preserve_labeled=True, skip_large_dirs=True):
        self.video_path = video_path
        self.video_name = Path(video_path).stem
        self.dry_run = dry_run
        self.preserve_labeled = preserve_labeled
        self.skip_large_dirs = skip_large_dirs
        
        # Statistics
        self.stats = {
            'total_crops_found': 0,
            'labeled_crops_found': 0,
            'unlabeled_crops_found': 0,
            'crops_to_delete': 0,
            'crops_preserved': 0,
            'embeddings_deleted': 0,
            'files_deleted': 0,
            'errors': 0,
            'skipped_dirs': 0
        }
        
        # Collect all relevant data
        self.labeled_images = {}
        self.video_crops = []
        self.crops_to_delete = []
        self.crops_to_preserve = []
        
    def quick_directory_check(self, directory):
        """Quick check of directory size to avoid scanning huge dirs."""
        if not directory.exists():
            return 0, False
            
        print(f"📏 Checking size of {directory}...")
        
        # Count files in top level first
        try:
            top_level_files = len([f for f in directory.iterdir() if f.is_file() and f.suffix.lower() == '.jpg'])
            
            # Count subdirectories
            subdirs = [d for d in directory.iterdir() if d.is_dir()]
            
            if len(subdirs) > 100:
                print(f"  📁 {directory} has {len(subdirs)} subdirectories - this could be slow!")
                
                if self.skip_large_dirs:
                    response = input(f"  Skip scanning {directory} (has {len(subdirs)} subdirs)? (Y/n): ")
                    if response.lower() != 'n':
                        print(f"  ⏭️ Skipping {directory}")
                        return 0, True
                        
            # Quick sample of a few subdirs to estimate total
            sample_size = min(5, len(subdirs))
            total_estimate = top_level_files
            
            if subdirs and sample_size > 0:
                sample_counts = []
                for subdir in subdirs[:sample_size]:
                    try:
                        count = len([f for f in subdir.iterdir() if f.is_file() and f.suffix.lower() == '.jpg'])
                        sample_counts.append(count)
                    except:
                        sample_counts.append(0)
                
                if sample_counts:
                    avg_per_subdir = sum(sample_counts) / len(sample_counts)
                    total_estimate += int(avg_per_subdir * len(subdirs))
            
            print(f"  📊 Estimated {total_estimate:,} JPG files in {directory}")
            
            if total_estimate > 50000 and self.skip_large_dirs:
                response = input(f"  Large directory (~{total_estimate:,} files). Skip? (Y/n): ")
                if response.lower() != 'n':
                    print(f"  ⏭️ Skipping large directory {directory}")
                    return total_estimate, True
                    
            return total_estimate, False
            
        except Exception as e:
            print(f"  ❌ Error checking {directory}: {e}")
            return 0, True
    
    def analyze_video_crops(self):
        """Analyze crops with smart directory handling."""
        print(f"🎯 Targeted analysis for video: {self.video_name}")
        
        # Get all labeled images for quick lookup
        if self.preserve_labeled:
            print("📥 Loading labeled images from database...")
            labeled_images_list = get_all_labeled_images()
            self.labeled_images = {img['image_path']: img for img in labeled_images_list}
            print(f"✅ Loaded {len(self.labeled_images)} labeled images")
        
        # Find crops with smart scanning
        self._smart_find_crops()
        
        # Categorize crops
        self._categorize_crops()
        
        # Log statistics
        self._log_analysis_stats()
        
    def _smart_find_crops(self):
        """Smart file scanning that handles huge directories."""
        crop_directories = [
            Path("videos"),           # Usually smaller, organized by video
            Path("processing"),       # Usually smaller  
            Path("non_crow_crops"),   # Usually smaller
            Path("potential_not_crow_crops"),  # Usually smaller
            Path("hard_negatives"),   # Usually smaller
            Path("false_positive_crops"),  # Usually smaller
            Path("crow_crops2"),      # Secondary - might be smaller
            Path("crow_crops"),       # Main one - often huge, scan last
        ]
        
        # Clean video name patterns
        clean_video_name = self.video_name.replace('.', '_').replace(' ', '_').lower()
        patterns = [clean_video_name, self.video_name.lower()]
        
        print(f"🔍 Searching for patterns: {patterns}")
        
        for crop_dir in crop_directories:
            if not crop_dir.exists():
                print(f"⏭️ {crop_dir} doesn't exist, skipping")
                continue
            
            # Check directory size first
            estimated_size, should_skip = self.quick_directory_check(crop_dir)
            
            if should_skip:
                self.stats['skipped_dirs'] += 1
                continue
                
            print(f"🔎 Scanning {crop_dir}...")
            found_in_dir = 0
            
            # Use targeted search patterns
            for pattern in patterns:
                try:
                    # More specific patterns to reduce search space
                    search_patterns = [
                        str(crop_dir / f"*{pattern}*.jpg"),  # Direct files
                        str(crop_dir / "*" / f"*{pattern}*.jpg"),  # One level deep
                    ]
                    
                    # For crow_crops, also try the organized structure
                    if crop_dir.name == "crow_crops":
                        search_patterns.append(str(crop_dir / "*" / f"*{pattern}*" / "*.jpg"))
                    
                    for search_pattern in search_patterns:
                        print(f"  🔍 Pattern: {search_pattern}")
                        matching_files = glob.glob(search_pattern, recursive=False)
                        
                        if matching_files:
                            print(f"    ✅ Found {len(matching_files)} files")
                            found_in_dir += len(matching_files)
                            
                            for file_path in tqdm(matching_files, desc=f"Processing {pattern}", leave=False):
                                self._add_crop_if_valid(file_path, crop_dir)
                        else:
                            print(f"    ⭕ No matches")
                            
                except Exception as e:
                    print(f"    ❌ Error with pattern {pattern}: {e}")
                    continue
            
            print(f"  📊 Total found in {crop_dir}: {found_in_dir}")
                        
        print(f"🎯 Total crops found: {len(self.video_crops)}")
    
    def _add_crop_if_valid(self, file_path, crop_dir):
        """Add crop if it's valid and not a duplicate."""
        image_file = Path(file_path)
        
        # Double-check the match
        if self._is_video_related_crop(image_file.name):
            crop_info = {
                'file_path': str(image_file),
                'posix_path': image_file.as_posix(),
                'filename': image_file.name,
                'directory': str(crop_dir),
                'size_bytes': image_file.stat().st_size if image_file.exists() else 0,
                'modified_time': datetime.fromtimestamp(image_file.stat().st_mtime) if image_file.exists() else None
            }
            
            # Avoid duplicates
            if not any(c['file_path'] == str(image_file) for c in self.video_crops):
                self.video_crops.append(crop_info)
    
    def _is_video_related_crop(self, filename):
        """Check if filename matches our video."""
        clean_video_name = self.video_name.replace('.', '_').replace(' ', '_').lower()
        clean_filename = filename.lower()
        
        return (clean_video_name in clean_filename or 
                self.video_name.lower() in clean_filename)
    
    def _categorize_crops(self):
        """Categorize crops into delete/preserve."""
        self.stats['total_crops_found'] = len(self.video_crops)
        
        if not self.video_crops:
            print("❌ No crops found for this video!")
            return
            
        print(f"📋 Categorizing {len(self.video_crops)} crops...")
        
        for crop in tqdm(self.video_crops, desc="Categorizing"):
            posix_path = crop['posix_path']
            
            if self.preserve_labeled and posix_path in self.labeled_images:
                label_info = self.labeled_images[posix_path]
                crop['label'] = label_info['label']
                crop['label_confidence'] = label_info.get('confidence')
                crop['is_training_data'] = label_info.get('is_training_data', False)
                
                self.crops_to_preserve.append(crop)
                self.stats['labeled_crops_found'] += 1
            else:
                self.crops_to_delete.append(crop)
                self.stats['unlabeled_crops_found'] += 1
        
        self.stats['crops_to_delete'] = len(self.crops_to_delete)
        self.stats['crops_preserved'] = len(self.crops_to_preserve)
    
    def _log_analysis_stats(self):
        """Log detailed analysis statistics."""
        print("\n" + "="*60)
        print("📊 TARGETED ANALYSIS SUMMARY")
        print("="*60)
        print(f"🎯 Video: {self.video_name}")
        print(f"📁 Directories scanned: {8 - self.stats['skipped_dirs']}")
        print(f"⏭️ Directories skipped: {self.stats['skipped_dirs']}")
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
        
        if self.crops_to_delete:
            print(f"\n🗑️ CROPS TO DELETE BY DIRECTORY:")
            dir_counts = Counter(crop['directory'] for crop in self.crops_to_delete)
            for directory, count in dir_counts.most_common():
                print(f"  {directory}: {count} images")
                
        if self.stats['skipped_dirs'] > 0:
            print(f"\n⚠️ Note: {self.stats['skipped_dirs']} directories were skipped to avoid long scanning times")
    
    def delete_database_embeddings(self):
        """Delete embeddings from database."""
        if not self.crops_to_delete:
            print("No crops to delete - skipping embedding deletion")
            return
            
        print(f"🗄️ Deleting database embeddings for {len(self.crops_to_delete)} crops...")
        
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
    
    def delete_crop_files(self):
        """Delete the actual crop image files."""
        if not self.crops_to_delete:
            print("No crop files to delete")
            return
            
        print(f"🗑️ Deleting {len(self.crops_to_delete)} crop files...")
        
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
    
    def create_backup_list(self):
        """Create backup list."""
        if not self.crops_to_delete:
            return
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = Path(f"deleted_crops_backup_{self.video_name}_{timestamp}.json")
        
        backup_data = {
            'video_name': self.video_name,
            'video_path': self.video_path,
            'deletion_timestamp': timestamp,
            'dry_run': self.dry_run,
            'preserve_labeled': self.preserve_labeled,
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
        print("🎯 TARGETED VIDEO CROP DELETION")
        print("=" * 60)
        
        self.analyze_video_crops()
        
        if not self.video_crops:
            print("❌ No crops found for this video. Check the video name or patterns.")
            return
        
        self.create_backup_list()
        
        if not self.dry_run and self.crops_to_delete:
            response = input(f"\n⚠️ Delete {len(self.crops_to_delete)} crop files and embeddings? (y/N): ")
            if response.lower() != 'y':
                print("❌ Deletion cancelled")
                return
        
        self.delete_database_embeddings()
        self.delete_crop_files()
        
        print("\n" + "="*60)
        print("🏁 DELETION COMPLETE")
        print("=" * 60)
        print(f"🎯 Video: {self.video_name}")
        print(f"🔧 Mode: {'DRY RUN' if self.dry_run else 'LIVE DELETION'}")
        print(f"🗑️ Files deleted: {self.stats['files_deleted']}")
        print(f"🗄️ Embeddings deleted: {self.stats['embeddings_deleted']}")
        print(f"💾 Files preserved: {self.stats['crops_preserved']}")
        print(f"❌ Errors: {self.stats['errors']}")

def main():
    parser = argparse.ArgumentParser(description="Targeted video crop deletion utility")
    parser.add_argument('video_path', help='Path to video file or video name')
    parser.add_argument('--execute', action='store_true', help='Execute the deletion')
    parser.add_argument('--no-preserve-labeled', action='store_true', help='Delete labeled images too')
    parser.add_argument('--scan-all', action='store_true', help='Scan all directories including huge ones')
    
    args = parser.parse_args()
    
    dry_run = not args.execute
    preserve_labeled = not args.no_preserve_labeled
    skip_large_dirs = not args.scan_all
    
    deletor = TargetedVideoCropDeletor(
        video_path=args.video_path,
        dry_run=dry_run,
        preserve_labeled=preserve_labeled,
        skip_large_dirs=skip_large_dirs
    )
    
    try:
        deletor.execute_deletion()
    except KeyboardInterrupt:
        print("\n❌ Deletion cancelled by user")

if __name__ == "__main__":
    main() 