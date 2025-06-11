#!/usr/bin/env python3
"""
Fast version of video crop deletion utility with progress indicators.
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
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FastVideoCropDeletor:
    """Fast version with progress indicators and efficient scanning."""
    
    def __init__(self, video_path, dry_run=True, preserve_labeled=True):
        self.video_path = video_path
        self.video_name = Path(video_path).stem
        self.dry_run = dry_run
        self.preserve_labeled = preserve_labeled
        
        # Statistics
        self.stats = {
            'total_crops_found': 0,
            'labeled_crops_found': 0,
            'unlabeled_crops_found': 0,
            'crops_to_delete': 0,
            'crops_preserved': 0,
            'embeddings_deleted': 0,
            'files_deleted': 0,
            'errors': 0
        }
        
        # Collect all relevant data
        self.labeled_images = {}
        self.video_crops = []
        self.crops_to_delete = []
        self.crops_to_preserve = []
        
    def analyze_video_crops(self):
        """Analyze all crops related to the video with progress."""
        logger.info(f"🔍 Analyzing crops for video: {self.video_name}")
        
        # Get all labeled images for quick lookup
        if self.preserve_labeled:
            print("Loading labeled images from database...")
            labeled_images_list = get_all_labeled_images()
            self.labeled_images = {img['image_path']: img for img in labeled_images_list}
            logger.info(f"Loaded {len(self.labeled_images)} labeled images from database")
        
        # Find crops in various locations
        self._find_crops_efficiently()
        
        # Categorize crops
        self._categorize_crops()
        
        # Log statistics
        self._log_analysis_stats()
        
    def _find_crops_efficiently(self):
        """Fast file scanning with progress indicators."""
        crop_directories = [
            Path("crow_crops"),
            Path("crow_crops2"), 
            Path("videos"),
            Path("processing"),
            Path("non_crow_crops"),
            Path("potential_not_crow_crops"),
            Path("hard_negatives"),
            Path("false_positive_crops")
        ]
        
        # Clean video name patterns for matching
        clean_video_name = self.video_name.replace('.', '_').replace(' ', '_').lower()
        patterns = [
            clean_video_name,
            self.video_name.lower(),
            Path(self.video_path).name.lower(),
        ]
        
        print(f"Searching for crops matching patterns: {patterns}")
        
        for crop_dir in crop_directories:
            if not crop_dir.exists():
                continue
                
            print(f"📁 Scanning {crop_dir}...")
            
            # Use faster glob patterns instead of rglob
            for pattern in patterns:
                # Try multiple glob patterns for efficiency
                glob_patterns = [
                    str(crop_dir / "**" / f"*{pattern}*.jpg"),
                    str(crop_dir / "*" / f"*{pattern}*.jpg"),
                    str(crop_dir / f"*{pattern}*.jpg")
                ]
                
                for glob_pattern in glob_patterns:
                    try:
                        matching_files = glob.glob(glob_pattern, recursive=True)
                        if matching_files:
                            print(f"  Found {len(matching_files)} files with pattern {pattern}")
                            
                            for file_path in tqdm(matching_files, desc=f"Processing {pattern}", leave=False):
                                image_file = Path(file_path)
                                
                                # Double-check the match to avoid false positives
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
                    except Exception as e:
                        print(f"  Error with pattern {glob_pattern}: {e}")
                        continue
                        
        logger.info(f"Found {len(self.video_crops)} total video-related crops")
    
    def _is_video_related_crop(self, filename):
        """Check if a filename is related to our target video."""
        clean_video_name = self.video_name.replace('.', '_').replace(' ', '_').lower()
        clean_filename = filename.lower()
        
        patterns = [
            clean_video_name,
            self.video_name.lower(),
            Path(self.video_path).name.lower(),
        ]
        
        return any(pattern in clean_filename for pattern in patterns)
    
    def _categorize_crops(self):
        """Categorize crops into those to delete and those to preserve."""
        self.stats['total_crops_found'] = len(self.video_crops)
        
        print(f"Categorizing {len(self.video_crops)} crops...")
        
        for crop in tqdm(self.video_crops, desc="Categorizing crops"):
            posix_path = crop['posix_path']
            
            # Check if crop is labeled
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
        print("📊 ANALYSIS SUMMARY")
        print("="*60)
        print(f"Video: {self.video_name}")
        print(f"Total crops found: {self.stats['total_crops_found']}")
        print(f"Labeled crops: {self.stats['labeled_crops_found']}")
        print(f"Unlabeled crops: {self.stats['unlabeled_crops_found']}")
        print(f"Crops to delete: {self.stats['crops_to_delete']}")
        print(f"Crops to preserve: {self.stats['crops_preserved']}")
        
        if self.crops_to_preserve:
            print("\n📋 PRESERVED CROPS (Labeled):")
            label_counts = Counter(crop.get('label', 'unknown') for crop in self.crops_to_preserve)
            for label, count in label_counts.most_common():
                print(f"  {label}: {count} images")
        
        if self.crops_to_delete:
            print(f"\n🗑️ CROPS TO DELETE:")
            dir_counts = Counter(crop['directory'] for crop in self.crops_to_delete)
            for directory, count in dir_counts.most_common():
                print(f"  {directory}: {count} images")
    
    def delete_database_embeddings(self):
        """Delete embeddings from database for crops that will be deleted."""
        if not self.crops_to_delete:
            print("No crops to delete - skipping embedding deletion")
            return
            
        print(f"🗄️ Deleting database embeddings for {len(self.crops_to_delete)} crops...")
        
        # Get embedding IDs for crops to delete
        crop_paths = [crop['file_path'] for crop in self.crops_to_delete]
        
        # Process in batches to avoid overwhelming the database
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
        """Create a backup list of deleted files for recovery."""
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
        """Execute the complete deletion process."""
        print("🚀 STARTING FAST VIDEO CROP DELETION")
        print("=" * 60)
        
        # Analyze what we're dealing with
        self.analyze_video_crops()
        
        # Create backup list
        self.create_backup_list()
        
        # Confirm deletion if not dry run
        if not self.dry_run and self.crops_to_delete:
            response = input(f"\n⚠️  This will delete {len(self.crops_to_delete)} crop files and their database entries. Continue? (y/N): ")
            if response.lower() != 'y':
                print("❌ Deletion cancelled by user")
                return
        
        # Delete database embeddings first
        self.delete_database_embeddings()
        
        # Delete crop files
        self.delete_crop_files()
        
        # Final summary
        self._log_final_summary()
    
    def _log_final_summary(self):
        """Log final deletion summary."""
        print("\n" + "="*60)
        print("🏁 DELETION COMPLETE")
        print("=" * 60)
        print(f"Video: {self.video_name}")
        print(f"Mode: {'DRY RUN' if self.dry_run else 'LIVE DELETION'}")
        print(f"Files deleted: {self.stats['files_deleted']}")
        print(f"Embeddings deleted: {self.stats['embeddings_deleted']}")
        print(f"Files preserved: {self.stats['crops_preserved']}")
        print(f"Errors: {self.stats['errors']}")

def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description="Fast video crop deletion utility")
    parser.add_argument('video_path', help='Path to video file or video name')
    parser.add_argument('--execute', action='store_true', help='Execute the deletion')
    parser.add_argument('--no-preserve-labeled', action='store_true', help='Delete labeled images too')
    
    args = parser.parse_args()
    
    dry_run = not args.execute
    preserve_labeled = not args.no_preserve_labeled
    
    deletor = FastVideoCropDeletor(
        video_path=args.video_path,
        dry_run=dry_run,
        preserve_labeled=preserve_labeled
    )
    
    try:
        deletor.execute_deletion()
    except KeyboardInterrupt:
        print("\n❌ Deletion cancelled by user")
    except Exception as e:
        print(f"❌ Error during deletion: {e}")
        raise

if __name__ == "__main__":
    main() 