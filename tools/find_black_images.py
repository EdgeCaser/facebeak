#!/usr/bin/env python3
"""
Simple script to identify black screen images for review.
This is a safe analysis-only script that won't modify anything.
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path
import logging

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze_image(image_path: str):
    """Analyze an image and return its statistics."""
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None, None, None, "Could not load image"
            
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Calculate statistics
        mean_val = np.mean(gray)
        std_val = np.std(gray)
        min_val = np.min(gray)
        max_val = np.max(gray)
        
        return mean_val, std_val, (min_val, max_val), None
        
    except Exception as e:
        return None, None, None, str(e)

def main():
    # Search in common directories
    search_dirs = [
        "crow_crops",
        "backup_labeled_crops", 
        "backup_all_labeled_crops",
        "non_crow_backup",
        "test_crops"
    ]
    
    all_black_images = []
    total_images = 0
    
    for search_dir in search_dirs:
        if not os.path.exists(search_dir):
            continue
            
        logger.info(f"Scanning directory: {search_dir}")
        
        # Find all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        
        for root, dirs, files in os.walk(search_dir):
            for file in files:
                if Path(file).suffix.lower() in image_extensions:
                    image_path = os.path.join(root, file)
                    total_images += 1
                    
                    mean_val, std_val, minmax, error = analyze_image(image_path)
                    
                    if error:
                        logger.warning(f"Error analyzing {image_path}: {error}")
                        continue
                    
                    # Check if it's likely a black image
                    # Using generous thresholds to catch various types of black screens
                    if mean_val < 15.0 and std_val < 8.0:  # Very dark with low variation
                        file_size = os.path.getsize(image_path)
                        all_black_images.append({
                            'path': image_path,
                            'mean': mean_val,
                            'std': std_val,
                            'minmax': minmax,
                            'size_bytes': file_size
                        })
                        logger.info(f"POTENTIAL BLACK IMAGE: {image_path}")
                        logger.info(f"  Mean: {mean_val:.2f}, Std: {std_val:.2f}, Range: {minmax}, Size: {file_size:,} bytes")
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print(f"Total images analyzed: {total_images}")
    print(f"Potential black images found: {len(all_black_images)}")
    
    if all_black_images:
        print(f"\nPOTENTIAL BLACK IMAGES:")
        print(f"{'='*60}")
        total_size = 0
        for img_info in all_black_images:
            print(f"Path: {img_info['path']}")
            print(f"  Stats: mean={img_info['mean']:.2f}, std={img_info['std']:.2f}")
            print(f"  Range: {img_info['minmax']}, Size: {img_info['size_bytes']:,} bytes")
            print()
            total_size += img_info['size_bytes']
        
        print(f"Total size of potential black images: {total_size:,} bytes ({total_size/1024/1024:.2f} MB)")
        
        # Save list to file for reference
        with open("potential_black_images.txt", "w") as f:
            f.write(f"Potential black images found on {os.path.basename(os.getcwd())}\n")
            f.write(f"Analysis date: {os.popen('date').read().strip()}\n")
            f.write(f"Total images analyzed: {total_images}\n")
            f.write(f"Potential black images: {len(all_black_images)}\n\n")
            
            for img_info in all_black_images:
                f.write(f"{img_info['path']}\n")
                f.write(f"  mean={img_info['mean']:.2f}, std={img_info['std']:.2f}, range={img_info['minmax']}, size={img_info['size_bytes']} bytes\n\n")
        
        print(f"\nList saved to: potential_black_images.txt")
        print(f"\nTo remove these images, you can use:")
        print(f"python tools/remove_black_images.py --dry-run")
    else:
        print("\nNo potential black images found!")

if __name__ == "__main__":
    main() 