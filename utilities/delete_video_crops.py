#!/usr/bin/env python3
"""
Utility to safely delete all crops from a specific video while preserving labeled images.
Provides comprehensive safety checks and detailed logging.
"""

import os
import sys
import json
import logging
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime
import argparse
import shutil

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db import (
    get_connection, 
    get_all_labeled_images, 
    get_image_label,
    delete_crow_embeddings,
    get_embedding_ids_by_image_paths
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class VideoCropDeletor:
    """Safely delete video crops while preserving labeled images."""
    
    def __init__(self, video_path, dry_run=True, preserve_labeled=True):
        """
        Initialize the video crop deletor.
        
        Args:
            video_path (str): Path to the video file or video name
            dry_run (bool): If True, only simulate the deletion
            preserve_labeled (bool): If True, preserve images that have labels
        """
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
        """Analyze all crops related to the video."""
        logger.info(f"🔍 Analyzing crops for video: {self.video_name}")
        
        # Get all labeled images for quick lookup
        if self.preserve_labeled:
            labeled_images_list = get_all_labeled_images()
            self.labeled_images = {img['image_path']: img for img in labeled_images_list}
            logger.info(f"Loaded {len(self.labeled_images)} labeled images from database")
        
        # Find crops in various locations
        self._find_crops_in_directories()
        self._find_crops_in_metadata()
        
        # Categorize crops
        self._categorize_crops()
        
        # Log statistics
        self._log_analysis_stats()
        
    def _find_crops_in_directories(self):
        """Find crops in standard crop directories."""
        crop_directories = [
            Path("dataset/crows/generic"),
            Path("dataset/not_crow"),
            Path("dataset/not_crow/hard_negatives"),
            # Legacy folders below:
            Path("crow_crops"),
            Path("crow_crops2"), 
            Path("videos"),
            Path("processing"),
            Path("non_crow_crops"),
            Path("potential_not_crow_crops"),
            Path("hard_negatives"),
            Path("false_positive_crops")
        ]
        
        for crop_dir in crop_directories:
            if not crop_dir.exists():
                continue
                
            logger.info(f"Scanning directory: {crop_dir}")
            
            # Search recursively for images matching video name
            for image_file in crop_dir.rglob("*.jpg"):
                filename = image_file.name
                
                # Check if filename contains video name
                if self._is_video_related_crop(filename):
                    crop_info = {
                        'file_path': str(image_file),
                        'posix_path': image_file.as_posix(),
                        'filename': filename,
                        'directory': str(crop_dir),
                        'size_bytes': image_file.stat().st_size if image_file.exists() else 0,
                        'modified_time': datetime.fromtimestamp(image_file.stat().st_mtime) if image_file.exists() else None
                    }
                    self.video_crops.append(crop_info)
                    
        logger.info(f"Found {len(self.video_crops)} potential video-related crops")
    
    def _find_crops_in_metadata(self):
        """Find crops referenced in crop metadata files."""
        metadata_files = [
            Path("crop_metadata.json"),
            Path("processing/crop_metadata.json"),
            Path("videos/crop_metadata.json")
        ]
        
        for metadata_file in metadata_files:
            if not metadata_file.exists():
                continue
                
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    
                crops = metadata.get('crops', {})
                for crop_path, crop_info in crops.items():
                    video_name = crop_info.get('video', '')
                    video_path = crop_info.get('video_path', '')
                    
                    # Check if this crop is from our target video
                    if (self.video_name in video_name or 
                        self.video_name in video_path or
                        self.video_path in video_path):
                        
                        full_path = Path(crop_path)
                        if not full_path.is_absolute():
                            # Try to resolve relative path
                            full_path = metadata_file.parent / crop_path
                        
                        # Add to crops if not already found
                        if not any(c['file_path'] == str(full_path) for c in self.video_crops):
                            crop_info = {
                                'file_path': str(full_path),
                                'posix_path': full_path.as_posix(),
                                'filename': full_path.name,
                                'directory': 'metadata',
                                'size_bytes': full_path.stat().st_size if full_path.exists() else 0,
                                'modified_time': datetime.fromtimestamp(full_path.stat().st_mtime) if full_path.exists() else None,
                                'metadata_source': str(metadata_file)
                            }
                            self.video_crops.append(crop_info)
                            
            except (json.JSONDecodeError, KeyError, OSError) as e:
                logger.warning(f"Error reading metadata file {metadata_file}: {e}")
    
    def _is_video_related_crop(self, filename):
        """Check if a filename is related to our target video."""
        # Clean video name for comparison
        clean_video_name = self.video_name.replace('.', '_').replace(' ', '_').lower()
        clean_filename = filename.lower()
        
        # Check various patterns
        patterns = [
            clean_video_name,
            self.video_name.lower(),
            Path(self.video_path).name.lower(),
        ]
        
        return any(pattern in clean_filename for pattern in patterns)
    
    def _categorize_crops(self):
        """Categorize crops into those to delete and those to preserve."""
        self.stats['total_crops_found'] = len(self.video_crops)
        
        for crop in self.video_crops:
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
        logger.info("📊 ANALYSIS SUMMARY")
        logger.info("=" * 50)
        logger.info(f"Video: {self.video_name}")
        logger.info(f"Total crops found: {self.stats['total_crops_found']}")
        logger.info(f"Labeled crops: {self.stats['labeled_crops_found']}")
        logger.info(f"Unlabeled crops: {self.stats['unlabeled_crops_found']}")
        logger.info(f"Crops to delete: {self.stats['crops_to_delete']}")
        logger.info(f"Crops to preserve: {self.stats['crops_preserved']}")
        
        if self.crops_to_preserve:
            logger.info("\n📋 PRESERVED CROPS (Labeled):")
            label_counts = Counter(crop.get('label', 'unknown') for crop in self.crops_to_preserve)
            for label, count in label_counts.most_common():
                logger.info(f"  {label}: {count} images")
        
        if self.crops_to_delete:
            logger.info(f"\n🗑️ CROPS TO DELETE:")
            dir_counts = Counter(crop['directory'] for crop in self.crops_to_delete)
            for directory, count in dir_counts.most_common():
                logger.info(f"  {directory}: {count} images")
    
    def delete_database_embeddings(self):
        """Delete embeddings from database for crops that will be deleted."""
        if not self.crops_to_delete:
            logger.info("No crops to delete - skipping embedding deletion")
            return
            
        logger.info(f"🗄️ Deleting database embeddings for {len(self.crops_to_delete)} crops...")
        
        # Get embedding IDs for crops to delete
        crop_paths = [crop['file_path'] for crop in self.crops_to_delete]
        embedding_mapping = get_embedding_ids_by_image_paths(crop_paths)
        
        if not embedding_mapping:
            logger.info("No database embeddings found for these crops")
            return
        
        embedding_ids = list(embedding_mapping.values())
        logger.info(f"Found {len(embedding_ids)} embeddings to delete")
        
        if not self.dry_run:
            try:
                deleted_count = delete_crow_embeddings(embedding_ids)
                self.stats['embeddings_deleted'] = deleted_count
                logger.info(f"✅ Deleted {deleted_count} embeddings from database")
            except Exception as e:
                logger.error(f"❌ Error deleting embeddings: {e}")
                self.stats['errors'] += 1
        else:
            logger.info(f"🔍 DRY RUN: Would delete {len(embedding_ids)} embeddings")
    
    def delete_crop_files(self):
        """Delete the actual crop image files."""
        if not self.crops_to_delete:
            logger.info("No crop files to delete")
            return
            
        logger.info(f"🗑️ Deleting {len(self.crops_to_delete)} crop files...")
        
        deleted_count = 0
        for crop in self.crops_to_delete:
            file_path = Path(crop['file_path'])
            
            try:
                if file_path.exists():
                    if not self.dry_run:
                        file_path.unlink()
                        deleted_count += 1
                        logger.debug(f"Deleted: {file_path}")
                    else:
                        logger.debug(f"DRY RUN: Would delete {file_path}")
                        deleted_count += 1
                else:
                    logger.debug(f"File already missing: {file_path}")
                    
            except Exception as e:
                logger.error(f"Error deleting {file_path}: {e}")
                self.stats['errors'] += 1
        
        self.stats['files_deleted'] = deleted_count
        
        if not self.dry_run:
            logger.info(f"✅ Deleted {deleted_count} crop files")
        else:
            logger.info(f"🔍 DRY RUN: Would delete {deleted_count} crop files")
    
    def cleanup_empty_directories(self):
        """Remove empty directories after deletion."""
        if self.dry_run:
            logger.info("🔍 DRY RUN: Skipping directory cleanup")
            return
            
        logger.info("🧹 Cleaning up empty directories...")
        
        # Collect all parent directories of deleted crops
        parent_dirs = set()
        for crop in self.crops_to_delete:
            parent_dirs.add(Path(crop['file_path']).parent)
        
        # Remove empty directories (from deepest to shallowest)
        for parent_dir in sorted(parent_dirs, key=lambda p: len(p.parts), reverse=True):
            try:
                if parent_dir.exists() and not any(parent_dir.iterdir()):
                    parent_dir.rmdir()
                    logger.info(f"Removed empty directory: {parent_dir}")
            except OSError:
                # Directory not empty or other issue
                pass
    
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
            logger.info(f"📄 Created backup list: {backup_file}")
        except Exception as e:
            logger.error(f"Error creating backup list: {e}")
    
    def execute_deletion(self):
        """Execute the complete deletion process."""
        logger.info("🚀 STARTING VIDEO CROP DELETION")
        logger.info("=" * 50)
        
        # Analyze what we're dealing with
        self.analyze_video_crops()
        
        # Create backup list
        self.create_backup_list()
        
        # Confirm deletion if not dry run
        if not self.dry_run and self.crops_to_delete:
            response = input(f"\n⚠️  This will delete {len(self.crops_to_delete)} crop files and their database entries. Continue? (y/N): ")
            if response.lower() != 'y':
                logger.info("❌ Deletion cancelled by user")
                return
        
        # Delete database embeddings first
        self.delete_database_embeddings()
        
        # Delete crop files
        self.delete_crop_files()
        
        # Clean up empty directories
        self.cleanup_empty_directories()
        
        # Final summary
        self._log_final_summary()
    
    def _log_final_summary(self):
        """Log final deletion summary."""
        logger.info("\n🏁 DELETION COMPLETE")
        logger.info("=" * 50)
        logger.info(f"Video: {self.video_name}")
        logger.info(f"Mode: {'DRY RUN' if self.dry_run else 'LIVE DELETION'}")
        logger.info(f"Preserve labeled: {self.preserve_labeled}")
        logger.info(f"Files deleted: {self.stats['files_deleted']}")
        logger.info(f"Embeddings deleted: {self.stats['embeddings_deleted']}")
        logger.info(f"Files preserved: {self.stats['crops_preserved']}")
        logger.info(f"Errors: {self.stats['errors']}")
        
        if self.stats['errors'] > 0:
            logger.warning(f"⚠️  {self.stats['errors']} errors occurred during deletion")
        
        if self.preserve_labeled and self.stats['crops_preserved'] > 0:
            logger.info(f"✅ {self.stats['crops_preserved']} labeled images were preserved")

def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(
        description="Safely delete all crops from a video while preserving labeled images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run (safe preview)
  python delete_video_crops.py video123.mp4
  
  # Delete unlabeled crops, preserve labeled ones
  python delete_video_crops.py video123.mp4 --execute
  
  # Delete ALL crops from video (including labeled)
  python delete_video_crops.py video123.mp4 --execute --no-preserve-labeled
  
  # Dry run with specific video name
  python delete_video_crops.py "my_video_name" --dry-run
        """
    )
    
    parser.add_argument(
        'video_path',
        help='Path to video file or video name'
    )
    
    parser.add_argument(
        '--execute',
        action='store_true',
        help='Execute the deletion (default is dry run)'
    )
    
    parser.add_argument(
        '--no-preserve-labeled',
        action='store_true',
        help='Delete ALL crops including labeled ones'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Only simulate the deletion (default mode)'
    )
    
    args = parser.parse_args()
    
    # Determine mode
    dry_run = not args.execute or args.dry_run
    preserve_labeled = not args.no_preserve_labeled
    
    # Create and run deletor
    deletor = VideoCropDeletor(
        video_path=args.video_path,
        dry_run=dry_run,
        preserve_labeled=preserve_labeled
    )
    
    try:
        deletor.execute_deletion()
    except KeyboardInterrupt:
        logger.info("\n❌ Deletion cancelled by user")
    except Exception as e:
        logger.error(f"❌ Error during deletion: {e}")
        raise

if __name__ == "__main__":
    main() 