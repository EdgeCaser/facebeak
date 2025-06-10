#!/usr/bin/env python3
"""
Quick script to remove black screen images based on pixel analysis.
This removes images with very low mean values that are essentially black screens.
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path
import shutil

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def is_black_image_fast(image_path: str) -> tuple:
    """
    Quick check if an image is essentially black.
    Returns (is_black, mean_value, std_value, error)
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            return False, 0, 0, "Could not load"
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mean_val = np.mean(gray)
        std_val = np.std(gray)
        
        # Very strict criteria for black images based on your data
        # Mean < 5 and std < 3 are definitely black screens
        is_black = mean_val < 5.0 and std_val < 3.5
        
        return is_black, mean_val, std_val, None
        
    except Exception as e:
        return False, 0, 0, str(e)

def remove_black_images_from_directory(directory: str, backup_dir: str = None, dry_run: bool = False):
    """Remove black images from a directory."""
    
    if backup_dir and not dry_run:
        Path(backup_dir).mkdir(parents=True, exist_ok=True)
    
    removed_count = 0
    total_size_removed = 0
    processed_count = 0
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    print(f"Processing directory: {directory}")
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if Path(file).suffix.lower() in image_extensions:
                image_path = os.path.join(root, file)
                processed_count += 1
                
                if processed_count % 100 == 0:
                    print(f"  Processed {processed_count} images, removed {removed_count}")
                
                is_black, mean_val, std_val, error = is_black_image_fast(image_path)
                
                if error:
                    print(f"  Error with {image_path}: {error}")
                    continue
                
                if is_black:
                    file_size = os.path.getsize(image_path)
                    total_size_removed += file_size
                    
                    print(f"  BLACK IMAGE: {image_path}")
                    print(f"    Mean: {mean_val:.2f}, Std: {std_val:.2f}, Size: {file_size:,} bytes")
                    
                    if not dry_run:
                        # Backup if requested
                        if backup_dir:
                            backup_path = Path(backup_dir) / f"backup_{removed_count:04d}_{Path(file).name}"
                            try:
                                shutil.copy2(image_path, backup_path)
                            except Exception as e:
                                print(f"    Warning: Could not backup {image_path}: {e}")
                        
                        # Remove the file
                        try:
                            os.remove(image_path)
                            print(f"    REMOVED: {image_path}")
                        except Exception as e:
                            print(f"    ERROR removing {image_path}: {e}")
                            continue
                    else:
                        print(f"    Would remove (dry run)")
                    
                    removed_count += 1
    
    return removed_count, total_size_removed, processed_count

def main():
    print("Quick Black Image Remover")
    print("=" * 50)
    
    # Search directories based on your project structure
    search_dirs = [
        "crow_crops/videos",
        "crow_crops/crows", 
        "backup_labeled_crops",
        "backup_all_labeled_crops"
    ]
    
    total_removed = 0
    total_size = 0
    total_processed = 0
    
    # Ask for confirmation
    print("This will remove images with mean < 5.0 and std < 3.5 (very black images)")
    print("Found directories to check:")
    for search_dir in search_dirs:
        if os.path.exists(search_dir):
            print(f"  - {search_dir}")
    
    response = input("\nDo you want to proceed? (y/N): ")
    if response.lower() != 'y':
        print("Cancelled.")
        return
    
    # Ask about backup
    backup_response = input("Create backup before removal? (Y/n): ")
    backup_dir = "black_images_backup" if backup_response.lower() != 'n' else None
    
    # Ask about dry run
    dry_run_response = input("Dry run first (recommended)? (Y/n): ")
    dry_run = dry_run_response.lower() != 'n'
    
    if dry_run:
        print("\n" + "=" * 50)
        print("DRY RUN MODE - NO FILES WILL BE REMOVED")
        print("=" * 50)
    
    for search_dir in search_dirs:
        if not os.path.exists(search_dir):
            continue
            
        print(f"\n--- Processing {search_dir} ---")
        removed, size_removed, processed = remove_black_images_from_directory(
            search_dir, backup_dir, dry_run
        )
        
        total_removed += removed
        total_size += size_removed
        total_processed += processed
        
        print(f"  Directory summary: {removed} black images found, {size_removed:,} bytes")
    
    print(f"\n" + "=" * 50)
    print(f"FINAL SUMMARY")
    print(f"=" * 50)
    print(f"Total images processed: {total_processed}")
    print(f"Black images found: {total_removed}")
    print(f"Total size: {total_size:,} bytes ({total_size/1024/1024:.2f} MB)")
    
    if dry_run and total_removed > 0:
        print(f"\nTo actually remove these files, run this script again and choose 'n' for dry run.")
    elif not dry_run and total_removed > 0:
        print(f"\nSuccessfully removed {total_removed} black images!")
        if backup_dir:
            print(f"Backup copies saved to: {backup_dir}")

if __name__ == "__main__":
    main() 