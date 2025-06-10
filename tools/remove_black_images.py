#!/usr/bin/env python3
"""
Script to identify and remove black screen images from the database.

This script scans image files, identifies images that are essentially black screens,
and removes them from both the filesystem and database records.
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path
import sqlite3
import logging
from typing import List, Tuple
import argparse

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db import get_connection, DB_PATH
from db_security import secure_database_connection

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def is_black_image(image_path: str, max_mean_value: float = 10.0, min_std_threshold: float = 5.0) -> bool:
    """
    Determine if an image is essentially a black screen.
    
    Args:
        image_path: Path to the image file
        max_mean_value: Maximum mean pixel value to consider black (0-255)
        min_std_threshold: Minimum standard deviation to avoid uniform colors
        
    Returns:
        True if the image appears to be a black screen
    """
    try:
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            logger.warning(f"Could not load image: {image_path}")
            return False
            
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Calculate statistics
        mean_val = np.mean(gray)
        std_val = np.std(gray)
        
        # Check if it's essentially black
        is_black = mean_val < max_mean_value and std_val < min_std_threshold
        
        if is_black:
            logger.info(f"Black image detected: {image_path} (mean: {mean_val:.2f}, std: {std_val:.2f})")
            
        return is_black
        
    except Exception as e:
        logger.error(f"Error analyzing image {image_path}: {e}")
        return False

def find_black_images_in_directory(directory_path: str, 
                                   max_mean_value: float = 10.0,
                                   min_std_threshold: float = 5.0) -> List[str]:
    """
    Find all black images in a directory.
    
    Args:
        directory_path: Path to search for images
        max_mean_value: Maximum mean pixel value to consider black
        min_std_threshold: Minimum standard deviation threshold
        
    Returns:
        List of paths to black images
    """
    black_images = []
    directory = Path(directory_path)
    
    if not directory.exists():
        logger.warning(f"Directory does not exist: {directory_path}")
        return black_images
    
    # Find all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_files = []
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if Path(file).suffix.lower() in image_extensions:
                image_files.append(os.path.join(root, file))
    
    logger.info(f"Analyzing {len(image_files)} images in {directory_path}")
    
    # Analyze each image
    for image_path in image_files:
        if is_black_image(image_path, max_mean_value, min_std_threshold):
            black_images.append(image_path)
    
    return black_images

def remove_image_from_database(image_path: str) -> bool:
    """
    Remove an image record from the database.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        True if successfully removed from database
    """
    try:
        conn = get_connection()
        c = conn.cursor()
        
        # Convert to consistent path format
        image_path_posix = Path(image_path).as_posix()
        
        # Check if image exists in image_labels table
        c.execute('SELECT id FROM image_labels WHERE image_path = ?', (image_path_posix,))
        label_record = c.fetchone()
        
        # Check if image is referenced in crow_embeddings table
        c.execute('SELECT id FROM crow_embeddings WHERE video_path LIKE ?', (f'%{Path(image_path).name}%',))
        embedding_records = c.fetchall()
        
        # Remove from image_labels if exists
        if label_record:
            c.execute('DELETE FROM image_labels WHERE image_path = ?', (image_path_posix,))
            logger.info(f"Removed image label record for: {image_path}")
        
        # Note: We're being cautious with crow_embeddings - these typically reference video files
        # not individual crop images, so we won't automatically delete them
        if embedding_records:
            logger.warning(f"Found {len(embedding_records)} embedding records that might reference this image: {image_path}")
            logger.warning("Manual review recommended for crow_embeddings table")
        
        conn.commit()
        conn.close()
        
        return True
        
    except Exception as e:
        logger.error(f"Error removing image from database: {image_path}: {e}")
        return False

def remove_image_file(image_path: str, backup_dir: str = None) -> bool:
    """
    Remove an image file, optionally backing it up first.
    
    Args:
        image_path: Path to the image file
        backup_dir: Optional directory to backup the file before deletion
        
    Returns:
        True if successfully removed
    """
    try:
        if backup_dir:
            # Create backup directory if it doesn't exist
            backup_path = Path(backup_dir)
            backup_path.mkdir(parents=True, exist_ok=True)
            
            # Create backup
            backup_file = backup_path / Path(image_path).name
            import shutil
            shutil.copy2(image_path, backup_file)
            logger.info(f"Backed up to: {backup_file}")
        
        # Remove the original file
        os.remove(image_path)
        logger.info(f"Removed file: {image_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error removing file {image_path}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Remove black screen images from database")
    parser.add_argument("--directory", "-d", default="crow_crops", 
                       help="Directory to scan for black images (default: crow_crops)")
    parser.add_argument("--max-mean", type=float, default=10.0,
                       help="Maximum mean pixel value to consider black (default: 10.0)")
    parser.add_argument("--min-std", type=float, default=5.0,
                       help="Minimum standard deviation threshold (default: 5.0)")
    parser.add_argument("--backup-dir", default="black_images_backup",
                       help="Directory to backup black images before deletion")
    parser.add_argument("--dry-run", action="store_true",
                       help="Only identify black images, don't remove them")
    parser.add_argument("--skip-backup", action="store_true",
                       help="Skip backing up images before deletion")
    
    args = parser.parse_args()
    
    logger.info(f"Scanning for black images in: {args.directory}")
    logger.info(f"Parameters: max_mean={args.max_mean}, min_std={args.min_std}")
    
    # Find black images
    black_images = find_black_images_in_directory(
        args.directory,
        max_mean_value=args.max_mean,
        min_std_threshold=args.min_std
    )
    
    if not black_images:
        logger.info("No black images found!")
        return
    
    logger.info(f"Found {len(black_images)} black images:")
    for img_path in black_images:
        logger.info(f"  {img_path}")
    
    if args.dry_run:
        logger.info("Dry run mode - no images will be removed")
        return
    
    # Confirm deletion
    response = input(f"\nDo you want to remove these {len(black_images)} black images? (y/N): ")
    if response.lower() != 'y':
        logger.info("Cancelled by user")
        return
    
    # Remove images
    removed_count = 0
    for image_path in black_images:
        logger.info(f"Processing: {image_path}")
        
        # Remove from database first
        db_success = remove_image_from_database(image_path)
        
        # Remove file (with optional backup)
        file_success = remove_image_file(
            image_path, 
            backup_dir=None if args.skip_backup else args.backup_dir
        )
        
        if db_success and file_success:
            removed_count += 1
        else:
            logger.error(f"Failed to fully remove: {image_path}")
    
    logger.info(f"Successfully removed {removed_count} of {len(black_images)} black images")
    
    if not args.skip_backup and removed_count > 0:
        logger.info(f"Backup copies saved to: {args.backup_dir}")

if __name__ == "__main__":
    main() 