#!/usr/bin/env python3
"""
Crop Architecture Demo and Migration Utility

This script demonstrates the new video/frame-based crop architecture that prevents
training bias by organizing crops by video and frame instead of crow ID.

NEW ARCHITECTURE (Prevents Training Bias):
crow_crops/
├── videos/
│   ├── video1/
│   │   ├── frame_000123_crop_001.jpg
│   │   ├── frame_000456_crop_001.jpg
│   │   └── frame_000789_crop_001.jpg
│   └── video2/
│       ├── frame_000234_crop_001.jpg
│       └── frame_000567_crop_001.jpg
├── metadata/
│   ├── crow_tracking.json      # Crow identity tracking
│   └── crop_metadata.json     # Maps crops to crow IDs
└── crows/                      # Legacy directory (backward compatibility)

BENEFITS:
- Prevents training bias by not grouping crops by crow identity
- Maintains temporal/spatial organization for analysis
- Still allows tracking via metadata
- Better for machine learning training
"""

import os
import sys
import json
from pathlib import Path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from crow_tracking import CrowTracker
import argparse

def demonstrate_new_architecture(base_dir="crow_crops"):
    """Demonstrate the new video/frame-based architecture."""
    print("🎯 NEW CROP ARCHITECTURE DEMONSTRATION")
    print("="*50)
    
    # Initialize tracker with new architecture
    tracker = CrowTracker(base_dir)
    
    print(f"\n📁 Directory Structure:")
    print(f"   Base: {tracker.base_dir}")
    print(f"   Videos: {tracker.videos_dir}")
    print(f"   Legacy Crows: {tracker.crows_dir}")
    print(f"   Metadata: {tracker.metadata_dir}")
    
    # Check if videos directory exists and has content
    if tracker.videos_dir.exists():
        video_dirs = list(tracker.videos_dir.iterdir())
        if video_dirs:
            print(f"\n📹 Found {len(video_dirs)} video directories:")
            for video_dir in video_dirs[:5]:  # Show first 5
                if video_dir.is_dir():
                    crop_files = list(video_dir.glob("*.jpg"))
                    print(f"   {video_dir.name}: {len(crop_files)} crops")
                    
                    # Show sample filenames
                    for crop_file in crop_files[:3]:
                        print(f"     - {crop_file.name}")
                    if len(crop_files) > 3:
                        print(f"     ... and {len(crop_files) - 3} more")
        else:
            print(f"\n📹 No video directories found yet")
    
    # Show crop metadata stats
    if tracker.crop_metadata["crops"]:
        print(f"\n📊 Crop Metadata Statistics:")
        print(f"   Total crops tracked: {len(tracker.crop_metadata['crops'])}")
        
        # Count crops by video
        video_counts = {}
        crow_counts = {}
        for crop_path, metadata in tracker.crop_metadata["crops"].items():
            video = metadata["video"]
            crow_id = metadata["crow_id"]
            video_counts[video] = video_counts.get(video, 0) + 1
            crow_counts[crow_id] = crow_counts.get(crow_id, 0) + 1
        
        print(f"   Videos with crops: {len(video_counts)}")
        print(f"   Unique crows: {len(crow_counts)}")
        
        # Show top videos by crop count
        if video_counts:
            top_videos = sorted(video_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            print(f"\n   Top videos by crop count:")
            for video, count in top_videos:
                print(f"     {video}: {count} crops")
    else:
        print(f"\n📊 No crop metadata found yet")
    
    return tracker

def analyze_training_bias_prevention():
    """Analyze how the new architecture prevents training bias."""
    print("\n🧠 TRAINING BIAS PREVENTION ANALYSIS")
    print("="*50)
    
    print("OLD ARCHITECTURE PROBLEMS:")
    print("❌ Crops grouped by crow ID → model learns crow-specific features")
    print("❌ Temporal relationships lost → model can't learn motion patterns")
    print("❌ Spatial context lost → model can't learn environmental cues")
    print("❌ Unbalanced datasets → some crows over-represented")
    
    print("\nNEW ARCHITECTURE BENEFITS:")
    print("✅ Crops organized by video/frame → preserves temporal context")
    print("✅ No crow ID grouping → prevents identity-specific overfitting")
    print("✅ Spatial relationships maintained → better environmental learning")
    print("✅ Balanced sampling possible → equal representation across videos")
    print("✅ Metadata tracking → still allows identity analysis when needed")

def migrate_legacy_crops(base_dir="crow_crops", dry_run=True):
    """Migrate crops from old crow-ID-based structure to new video/frame-based structure."""
    print(f"\n🔄 LEGACY CROP MIGRATION {'(DRY RUN)' if dry_run else ''}")
    print("="*50)
    
    base_path = Path(base_dir)
    crows_dir = base_path / "crows"
    videos_dir = base_path / "videos"
    
    if not crows_dir.exists():
        print("❌ No legacy crows directory found")
        return
    
    # Find all legacy crop files
    legacy_crops = []
    for crow_dir in crows_dir.iterdir():
        if crow_dir.is_dir() and crow_dir.name.startswith("crow_"):
            for crop_file in crow_dir.glob("*.jpg"):
                legacy_crops.append((crow_dir.name, crop_file))
    
    print(f"📊 Found {len(legacy_crops)} legacy crop files")
    
    if not legacy_crops:
        print("✅ No legacy crops to migrate")
        return
    
    # Analyze legacy crop filenames to extract video/frame info
    migration_plan = []
    for crow_id, crop_file in legacy_crops:
        filename = crop_file.name
        
        # Try to extract video name and frame number from filename
        # Expected format: crop_XXXXXXXX_timestamp_videoname_frame_XXXXXX.jpg
        parts = filename.split('_')
        if len(parts) >= 6 and 'frame' in parts:
            try:
                frame_idx = parts.index('frame')
                if frame_idx + 1 < len(parts):
                    frame_num = int(parts[frame_idx + 1].split('.')[0])
                    video_name = '_'.join(parts[3:frame_idx])
                    
                    new_filename = f"frame_{frame_num:06d}_crop_001.jpg"
                    new_path = videos_dir / video_name / new_filename
                    
                    migration_plan.append({
                        'old_path': crop_file,
                        'new_path': new_path,
                        'crow_id': crow_id,
                        'video': video_name,
                        'frame': frame_num
                    })
            except (ValueError, IndexError):
                print(f"⚠️  Could not parse filename: {filename}")
    
    print(f"📋 Migration plan: {len(migration_plan)} files")
    
    if dry_run:
        print("\n🔍 DRY RUN - Would perform these migrations:")
        for i, plan in enumerate(migration_plan[:10]):  # Show first 10
            print(f"   {plan['old_path']} → {plan['new_path']}")
        if len(migration_plan) > 10:
            print(f"   ... and {len(migration_plan) - 10} more")
    else:
        print("\n🚀 Performing migration...")
        # TODO: Implement actual migration logic
        print("⚠️  Actual migration not implemented yet - use dry_run=True for now")

def main():
    parser = argparse.ArgumentParser(description="Crop Architecture Demo and Migration Utility")
    parser.add_argument("--base-dir", default="crow_crops", help="Base directory for crops")
    parser.add_argument("--migrate", action="store_true", help="Migrate legacy crops")
    parser.add_argument("--no-dry-run", action="store_true", help="Actually perform migration (not just dry run)")
    
    args = parser.parse_args()
    
    # Demonstrate new architecture
    tracker = demonstrate_new_architecture(args.base_dir)
    
    # Analyze bias prevention
    analyze_training_bias_prevention()
    
    # Migrate legacy crops if requested
    if args.migrate:
        migrate_legacy_crops(args.base_dir, dry_run=not args.no_dry_run)
    
    print(f"\n✨ SUMMARY")
    print("="*50)
    print("The new video/frame-based architecture prevents training bias by:")
    print("1. Organizing crops by video and frame (not crow ID)")
    print("2. Preserving temporal and spatial context")
    print("3. Enabling balanced sampling across videos")
    print("4. Maintaining crow identity tracking via metadata")
    print("\nThis leads to better machine learning models that generalize well!")

if __name__ == "__main__":
    main() 