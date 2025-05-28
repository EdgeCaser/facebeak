#!/usr/bin/env python3
"""
Quick test extraction with very low confidence thresholds.
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from detection import detect_crows_parallel
from crow_tracking import CrowTracker

def quick_test_video(video_path, output_dir="test_crops"):
    """Quick test extraction on a video with very low thresholds."""
    print(f"🧪 Quick test extraction on: {video_path}")
    print(f"Output directory: {output_dir}")
    
    if not os.path.exists(video_path):
        print(f"❌ Video not found: {video_path}")
        return
    
    # Initialize tracker
    tracker = CrowTracker(base_dir=output_dir)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Cannot open video: {video_path}")
        return
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Video info: {total_frames} frames, {fps} FPS")
    
    # Test on 20 frames from different parts of the video
    test_frames = []
    frame_numbers = []
    
    for i in range(20):
        frame_num = int(i * total_frames / 20)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if ret:
            test_frames.append(frame)
            frame_numbers.append(frame_num)
    
    cap.release()
    
    if not test_frames:
        print("❌ Could not read any frames")
        return
    
    print(f"Testing on {len(test_frames)} frames...")
    
    # Very low confidence thresholds
    confidence_levels = [0.05, 0.1, 0.2, 0.3]
    
    for confidence in confidence_levels:
        print(f"\n🎯 Testing with confidence threshold: {confidence}")
        
        try:
            detections = detect_crows_parallel(
                test_frames,
                score_threshold=confidence,
                yolo_threshold=confidence,  # Also lower YOLO threshold
                multi_view_yolo=False,
                multi_view_rcnn=False
            )
            
            total_detections = sum(len(d) for d in detections)
            print(f"   Total detections: {total_detections}")
            
            if total_detections > 0:
                print(f"   ✅ Found detections at confidence {confidence}!")
                
                # Process a few detections to test the full pipeline
                crops_saved = 0
                for i, frame_dets in enumerate(detections[:5]):  # Test first 5 frames
                    if frame_dets:
                        frame = test_frames[i]
                        frame_num = frame_numbers[i]
                        frame_time = frame_num / fps if fps > 0 else None
                        
                        for det in frame_dets[:2]:  # Test first 2 detections per frame
                            crow_id = tracker.process_detection(
                                frame, frame_num, det, video_path, frame_time
                            )
                            if crow_id:
                                crops_saved += 1
                                print(f"     Saved crop for {crow_id} (frame {frame_num})")
                
                print(f"   Crops saved: {crops_saved}")
                
                if crops_saved > 0:
                    print(f"\n✅ SUCCESS! Extraction working at confidence {confidence}")
                    print(f"Check the output directory: {output_dir}")
                    
                    # Show what was created
                    videos_dir = Path(output_dir) / "videos"
                    if videos_dir.exists():
                        for video_dir in videos_dir.iterdir():
                            if video_dir.is_dir():
                                images = list(video_dir.glob("*.jpg"))
                                print(f"   {video_dir.name}: {len(images)} images")
                    
                    return True
            else:
                print(f"   No detections at confidence {confidence}")
                
        except Exception as e:
            print(f"   ❌ Error at confidence {confidence}: {e}")
    
    print(f"\n❌ No detections found at any confidence level")
    print(f"This suggests your videos may not contain detectable birds/crows")
    return False

def main():
    if len(sys.argv) < 2:
        print("Usage: python quick_test_extraction.py <video_path> [output_dir]")
        print("Example: python quick_test_extraction.py /path/to/video.mp4")
        return
    
    video_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "test_crops"
    
    success = quick_test_video(video_path, output_dir)
    
    if success:
        print(f"\n🎉 Extraction is working! You can now run the full extraction with:")
        print(f"python utilities/extract_crops_cli.py <video_directory> --min-confidence 0.05 --output-dir {output_dir}")
    else:
        print(f"\n💡 Suggestions:")
        print(f"1. Try a different video that clearly shows birds")
        print(f"2. Check if your videos are outdoor scenes with birds visible")
        print(f"3. The models might not recognize the specific type of birds in your videos")
        print(f"4. Consider using the GUI tool for manual review: python utilities/extract_training_gui.py")

if __name__ == "__main__":
    main() 