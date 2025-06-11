#!/usr/bin/env python3
"""
Example script showing how to use the video crop deletion utility.
"""

import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utilities.delete_video_crops import VideoCropDeletor

def example_dry_run():
    """Example: Safe preview of what would be deleted."""
    print("=== DRY RUN EXAMPLE ===")
    
    # Replace with your actual video name/path
    video_path = "your_video_name_here.mp4"
    
    deletor = VideoCropDeletor(
        video_path=video_path,
        dry_run=True,           # Safe mode - only shows what would be deleted
        preserve_labeled=True   # Keep labeled images
    )
    
    deletor.execute_deletion()

def example_delete_unlabeled():
    """Example: Delete only unlabeled crops, preserve labeled ones."""
    print("=== DELETE UNLABELED CROPS EXAMPLE ===")
    
    video_path = "your_video_name_here.mp4"
    
    deletor = VideoCropDeletor(
        video_path=video_path,
        dry_run=False,          # Actually delete files
        preserve_labeled=True   # Keep labeled images
    )
    
    deletor.execute_deletion()

def example_delete_all():
    """Example: Delete ALL crops from video (including labeled ones)."""
    print("=== DELETE ALL CROPS EXAMPLE ===")
    
    video_path = "your_video_name_here.mp4"
    
    deletor = VideoCropDeletor(
        video_path=video_path,
        dry_run=False,           # Actually delete files
        preserve_labeled=False   # Delete labeled images too
    )
    
    deletor.execute_deletion()

def interactive_example():
    """Interactive example that asks user for video name."""
    print("=== INTERACTIVE EXAMPLE ===")
    
    video_name = input("Enter video name or path: ").strip()
    if not video_name:
        print("No video name provided. Exiting.")
        return
    
    mode = input("Mode (1=dry-run, 2=delete-unlabeled, 3=delete-all): ").strip()
    
    if mode == "1":
        dry_run = True
        preserve_labeled = True
        print("Mode: Dry run (safe preview)")
    elif mode == "2":
        dry_run = False
        preserve_labeled = True
        print("Mode: Delete unlabeled crops only")
    elif mode == "3":
        dry_run = False
        preserve_labeled = False
        print("Mode: Delete ALL crops")
    else:
        print("Invalid mode. Using dry run.")
        dry_run = True
        preserve_labeled = True
    
    deletor = VideoCropDeletor(
        video_path=video_name,
        dry_run=dry_run,
        preserve_labeled=preserve_labeled
    )
    
    deletor.execute_deletion()

if __name__ == "__main__":
    print("Video Crop Deletion Utility Examples")
    print("=" * 40)
    
    if len(sys.argv) > 1:
        example_type = sys.argv[1]
        
        if example_type == "dry-run":
            example_dry_run()
        elif example_type == "delete-unlabeled":
            example_delete_unlabeled()
        elif example_type == "delete-all":
            example_delete_all()
        elif example_type == "interactive":
            interactive_example()
        else:
            print(f"Unknown example type: {example_type}")
            print("Available examples: dry-run, delete-unlabeled, delete-all, interactive")
    else:
        print("Usage examples:")
        print("  python example_delete_video_crops.py dry-run")
        print("  python example_delete_video_crops.py delete-unlabeled")  
        print("  python example_delete_video_crops.py delete-all")
        print("  python example_delete_video_crops.py interactive")
        print("\nFor the main utility, use:")
        print("  python delete_video_crops.py your_video.mp4 --help") 