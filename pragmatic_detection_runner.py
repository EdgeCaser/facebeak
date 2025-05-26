#!/usr/bin/env python3
"""
Pragmatic Detection Runner - Handle Domain Mismatch Issues

This script acknowledges that your video environment has significant domain mismatch
with COCO training data, causing high-confidence false positives. It provides a
practical workflow to still build useful training data.

Strategy:
1. Use VERY strict detection settings
2. Process with expectation of false positives  
3. Use manual review tools to clean up
4. Build initial dataset despite noise
"""

import subprocess
import sys
import os
from pathlib import Path

def run_strict_detection(video_dir, output_dir="crow_crops_strict"):
    """
    Run detection with very strict settings optimized for your domain mismatch issues.
    """
    print("🔧 PRAGMATIC DETECTION STRATEGY")
    print("="*50)
    print("⚠️  Acknowledging domain mismatch issues")
    print("🎯 Using strict settings + manual cleanup workflow")
    print()
    
    # Very strict settings
    cmd = [
        sys.executable, 
        "extract_training_data.py",
        video_dir,
        "--min-confidence", "0.7",     # Very high threshold
        "--min-detections", "5",       # Only consistently detected crows
        "--batch-size", "32",          # RTX 3080 optimized
        "--output-dir", output_dir
    ]
    
    print("📊 STRICT SETTINGS:")
    print(f"   Min Confidence: 0.7 (very high)")
    print(f"   Min Detections: 5 (very strict)")
    print(f"   Output: {output_dir}")
    print()
    print("🎯 This will:")
    print("   ✅ Reduce false positives (but not eliminate)")
    print("   ✅ Only keep consistently detected crows")
    print("   ⚠️  May miss some real crows (acceptable trade-off)")
    print()
    
    response = input("Continue with strict detection? (y/n): ")
    if response.lower() != 'y':
        print("Cancelled.")
        return False
        
    print("🚀 Starting strict detection...")
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Detection failed: {e}")
        return False

def run_manual_cleanup(crop_dir="crow_crops_strict"):
    """
    Guide user through manual cleanup process.
    """
    print("\n🧹 MANUAL CLEANUP PHASE")
    print("="*30)
    
    if not os.path.exists(crop_dir):
        print(f"❌ Crop directory {crop_dir} not found")
        return
        
    print(f"📁 Processing crops in: {crop_dir}")
    print()
    print("🎯 CLEANUP STRATEGY:")
    print("   1. Use Image Reviewer to label false positives")
    print("   2. Mark multi-crow images") 
    print("   3. Build clean training dataset")
    print()
    print("🔧 TOOLS AVAILABLE:")
    print("   • Image Reviewer: Quick labeling (press 2 for false positives)")
    print("   • Suspect Lineup: Identity verification")
    print("   • Database cleanup: Automatic exclusion of marked items")
    print()
    
    response = input("Launch Image Reviewer for manual cleanup? (y/n): ")
    if response.lower() == 'y':
        print("🚀 Launching Image Reviewer...")
        try:
            subprocess.run([sys.executable, "image_reviewer.py"], check=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ Image Reviewer failed: {e}")

def check_results(crop_dir="crow_crops_strict"):
    """
    Analyze results and provide recommendations.
    """
    print("\n📊 RESULTS ANALYSIS")
    print("="*25)
    
    if not os.path.exists(crop_dir):
        print(f"❌ No results found in {crop_dir}")
        return
        
    # Count crop directories (each represents a detected crow)
    crop_dirs = [d for d in Path(crop_dir).iterdir() if d.is_dir() and d.name.startswith('crow_')]
    
    print(f"📈 DETECTION RESULTS:")
    print(f"   Total crow IDs detected: {len(crop_dirs)}")
    
    # Count total images
    total_images = 0
    for crow_dir in crop_dirs:
        image_files = list(crow_dir.glob("*.jpg")) + list(crow_dir.glob("*.png"))
        total_images += len(image_files)
    
    print(f"   Total crop images: {total_images}")
    print()
    
    if len(crop_dirs) == 0:
        print("⚠️  NO CROWS DETECTED")
        print("   This suggests your domain mismatch is severe.")
        print("   Consider:")
        print("   • Lower confidence threshold (0.6)")
        print("   • Different camera angles/lighting")
        print("   • Manual crop extraction")
        return False
        
    elif len(crop_dirs) < 10:
        print("⚠️  FEW CROWS DETECTED")
        print("   This might be too strict, but quality should be higher.")
        print("   Proceed with manual review.")
        
    else:
        print("✅ GOOD DETECTION COUNT")
        print("   Proceed with manual cleanup to build training set.")
    
    print()
    print("🎯 NEXT STEPS:")
    print("   1. Review crops manually (expect some false positives)")
    print("   2. Label false positives using Image Reviewer")
    print("   3. Train on cleaned dataset")
    print("   4. Use trained model for better future detection")
    
    return True

def main():
    """
    Main pragmatic detection workflow.
    """
    print("🎯 PRAGMATIC DETECTION WORKFLOW")
    print("For handling domain mismatch in crow detection")
    print("="*50)
    print()
    
    # Get video directory
    video_dir = input("Enter video directory path (or press Enter for 'videos/batch 1'): ").strip()
    if not video_dir:
        video_dir = "videos/batch 1"
    
    if not os.path.exists(video_dir):
        print(f"❌ Video directory '{video_dir}' not found")
        return
    
    print(f"📁 Processing videos from: {video_dir}")
    print()
    
    # Phase 1: Strict detection
    if not run_strict_detection(video_dir):
        return
    
    # Phase 2: Manual cleanup
    run_manual_cleanup()
    
    # Phase 3: Results analysis
    if check_results():
        print("\n🎉 READY FOR TRAINING!")
        print("Your cleaned dataset is ready for overnight training.")
        print("Use: python quick_start_training.py")

if __name__ == "__main__":
    main() 