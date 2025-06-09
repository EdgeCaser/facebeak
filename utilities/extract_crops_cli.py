#!/usr/bin/env python3
"""
CLI tool for extracting crow crops with full parameter control.
"""

import os
import sys
import cv2
import numpy as np
from tqdm import tqdm
import argparse

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detection import detect_crows_parallel
from crow_tracking import CrowTracker
from collections import defaultdict
from logging_config import setup_logging
from video_orientation import apply_video_orientation

# Configure logging
logger = setup_logging()

def extract_crops_from_video(video_path, tracker, min_confidence=0.2, min_detections=2, 
                           batch_size=32, multi_view_yolo=False, multi_view_rcnn=False, 
                           nms_threshold=0.3, apply_video_orientation_correction=True):
    """
    Extract crow crops from a video for training with full parameter control.
    """
    logger.info(f"Processing video: {video_path}")
    logger.info(f"Parameters: confidence={min_confidence}, min_detections={min_detections}")
    logger.info(f"Multi-view: YOLO={multi_view_yolo}, RCNN={multi_view_rcnn}")
    logger.info(f"NMS threshold: {nms_threshold}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Failed to open video: {video_path}")
        return
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    logger.info(f"Video info: {total_frames} frames, {fps} FPS")
    
    # First pass: collect all detections without saving crops
    all_detections = []  # Store (frame, frame_num, det, frame_time) tuples
    detections_by_crow = defaultdict(list)
    
    logger.info("First pass: Detecting crows and building tracks...")
    with tqdm(total=total_frames, desc="Detecting") as pbar:
        while True:
            frames = []
            frame_numbers = []
            for _ in range(batch_size):
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Apply video orientation correction if enabled
                if apply_video_orientation_correction:
                    frame = apply_video_orientation(frame, video_path)
                
                frames.append(frame)
                frame_numbers.append(int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1)
            if not frames:
                break
            
            # Call detection with all parameters
            detections = detect_crows_parallel(
                frames, 
                score_threshold=min_confidence,
                multi_view_yolo=multi_view_yolo,
                multi_view_rcnn=multi_view_rcnn,
                nms_threshold=nms_threshold
            )
            
            for frame_idx, frame_dets in enumerate(detections):
                frame = frames[frame_idx]
                frame_num = frame_numbers[frame_idx]
                frame_time = frame_num / fps if fps > 0 else None
                
                for det in frame_dets:
                    if det['score'] < min_confidence:
                        continue
                    
                    # Store detection for second pass
                    all_detections.append((frame.copy(), frame_num, det, frame_time))
            
            pbar.update(len(frames))
    
    cap.release()
    
    # Second pass: process detections and assign crow IDs
    logger.info("Second pass: Assigning crow IDs...")
    for frame, frame_num, det, frame_time in tqdm(all_detections, desc="Assigning IDs"):
        crow_id = tracker.process_detection(frame, frame_num, det, video_path, frame_time)
        if crow_id:
            detections_by_crow[crow_id].append((frame_num, det))
    
    # Third pass: save crops only for crows meeting minimum detections threshold
    logger.info("Third pass: Saving crops for qualifying crows...")
    qualifying_crows = {crow_id: dets for crow_id, dets in detections_by_crow.items() 
                       if len(dets) >= min_detections}
    
    crops_saved = 0
    for crow_id, crow_detections in qualifying_crows.items():
        logger.info(f"Saving crops for {crow_id} ({len(crow_detections)} detections)")
        # For qualifying crows, process a subset of their best detections
        # Sort by confidence and take up to 10 best detections
        sorted_dets = sorted(crow_detections, key=lambda x: x[1]['score'], reverse=True)[:10]
        
        for frame_num, det in sorted_dets:
            # Re-extract the frame for this detection
            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                # Force save crop by calling tracker's save_crop directly
                from tracking import extract_normalized_crow_crop
                crop = extract_normalized_crow_crop(frame, det['bbox'], correct_orientation=True, padding=0.3)
                if crop:
                    crop_path = tracker.save_crop(crop, crow_id, frame_num, video_path)
                    if crop_path:
                        crops_saved += 1
    
    # Log summary
    total_crows = len(detections_by_crow)
    crows_with_min_detections = len(qualifying_crows)
    logger.info(f"Found {total_crows} crows, {crows_with_min_detections} with {min_detections}+ detections")
    logger.info(f"Saved {crops_saved} crop files")
    
    return detections_by_crow

def find_video_files(video_dir, recursive=False):
    """
    Find video files in directory, optionally recursively.
    
    Args:
        video_dir: Directory to search for videos
        recursive: Whether to search subdirectories recursively
        
    Returns:
        List of video file paths
    """
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.MP4', '.AVI', '.MOV', '.MKV')
    video_files = []
    
    if recursive:
        # Use os.walk for recursive search
        for root, dirs, files in os.walk(video_dir):
            for file in files:
                if file.endswith(video_extensions):
                    video_files.append(os.path.join(root, file))
    else:
        # Non-recursive search (original behavior)
        try:
            for file in os.listdir(video_dir):
                if file.endswith(video_extensions):
                    video_files.append(os.path.join(video_dir, file))
        except OSError as e:
            logger.error(f"Error reading directory {video_dir}: {e}")
            return []
    
    return sorted(video_files)

def main():
    parser = argparse.ArgumentParser(description="Extract crow crops from videos with full parameter control")
    parser.add_argument("video_dir", help="Directory containing input videos")
    parser.add_argument("--output-dir", default="crow_crops", help="Base directory to save crops")
    parser.add_argument("--min-confidence", type=float, default=0.2, help="Minimum detection confidence")
    parser.add_argument("--min-detections", type=int, default=2, help="Minimum detections per crow")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for processing")
    parser.add_argument("--multi-view-yolo", action="store_true", help="Enable multi-view for YOLO")
    parser.add_argument("--multi-view-rcnn", action="store_true", help="Enable multi-view for R-CNN")
    parser.add_argument("--nms-threshold", type=float, default=0.3, help="NMS/IOU threshold for merging boxes")
    parser.add_argument("--recursive", action="store_true", help="Search for videos recursively in subdirectories")
    
    args = parser.parse_args()
    
    # Validate video directory
    if not os.path.isdir(args.video_dir):
        logger.error(f"Video directory does not exist: {args.video_dir}")
        sys.exit(1)
    
    # Initialize crow tracker
    tracker = CrowTracker(base_dir=args.output_dir)
    
    # Create processing run directory
    run_dir = tracker.create_processing_run()
    logger.info(f"Starting extraction with run directory: {run_dir}")
    logger.info(f"Parameters: confidence={args.min_confidence}, min_detections={args.min_detections}")
    logger.info(f"Multi-view: YOLO={args.multi_view_yolo}, RCNN={args.multi_view_rcnn}")
    logger.info(f"NMS threshold: {args.nms_threshold}")
    logger.info(f"Recursive search: {args.recursive}")
    
    # Find all video files
    video_files = find_video_files(args.video_dir, recursive=args.recursive)
    
    if not video_files:
        logger.warning(f"No video files found in {args.video_dir}")
        logger.info("Supported formats: .mp4, .avi, .mov, .mkv")
        sys.exit(0)
    
    logger.info(f"Found {len(video_files)} video files to process")
    if args.recursive:
        logger.info("Videos found in:")
        for video_file in video_files:
            rel_path = os.path.relpath(video_file, args.video_dir)
            logger.info(f"  {rel_path}")
    
    total_crows_before = len(tracker.list_crows())
    
    for video_file in video_files:
        rel_path = os.path.relpath(video_file, args.video_dir)
        logger.info(f"\nProcessing {rel_path}...")
        extract_crops_from_video(
            video_file,  # Use full path instead of joining again
            tracker,
            min_confidence=args.min_confidence,
            min_detections=args.min_detections,
            batch_size=args.batch_size,
            multi_view_yolo=args.multi_view_yolo,
            multi_view_rcnn=args.multi_view_rcnn,
            nms_threshold=args.nms_threshold
        )
    
    # Clean up processing directory
    tracker.cleanup_processing_dir(run_dir)
    
    # Log summary
    total_crows_after = len(tracker.list_crows())
    new_crows = total_crows_after - total_crows_before
    logger.info(f"\nExtraction complete:")
    logger.info(f"Total crows in database: {total_crows_after}")
    logger.info(f"New crows added: {new_crows}")
    
    if args.recursive and video_files:
        logger.info(f"Processed videos from {len(set(os.path.dirname(f) for f in video_files))} directories")

if __name__ == "__main__":
    main() 