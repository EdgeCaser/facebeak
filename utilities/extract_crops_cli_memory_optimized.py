#!/usr/bin/env python3
"""
Memory-optimized CLI tool for extracting crow crops with full parameter control.
This version avoids storing entire frames in memory to prevent OOM kills on EC2.
"""

import os
import sys
import cv2
import numpy as np
from tqdm import tqdm
import argparse
import gc

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detection import detect_crows_parallel
from crow_tracking import CrowTracker
from collections import defaultdict
from logging_config import setup_logging
from video_orientation import apply_video_orientation

# Configure logging
logger = setup_logging()

def extract_crops_from_video_memory_optimized(video_path, tracker, min_confidence=0.2, min_detections=2, 
                           batch_size=16, multi_view_yolo=False, multi_view_rcnn=False, 
                           nms_threshold=0.3, apply_video_orientation_correction=True):
    """
    Memory-optimized extraction that processes detections immediately without storing full frames.
    """
    logger.info(f"Processing video: {video_path}")
    logger.info(f"Parameters: confidence={min_confidence}, min_detections={min_detections}")
    logger.info(f"Multi-view: YOLO={multi_view_yolo}, RCNN={multi_view_rcnn}")
    logger.info(f"NMS threshold: {nms_threshold}, batch_size: {batch_size}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Failed to open video: {video_path}")
        return
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    logger.info(f"Video info: {total_frames} frames, {fps} FPS")
    
    # Track detections by crow ID without storing frames
    detections_by_crow = defaultdict(list)
    frame_detection_info = []  # Store minimal info: (frame_num, det, video_path, frame_time)
    
    logger.info("Pass 1: Detecting crows and assigning IDs...")
    with tqdm(total=total_frames, desc="Processing") as pbar:
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
            
            # Process detections immediately while frames are in memory
            for frame_idx, frame_dets in enumerate(detections):
                frame = frames[frame_idx]
                frame_num = frame_numbers[frame_idx]
                frame_time = frame_num / fps if fps > 0 else None
                
                for det in frame_dets:
                    if det['score'] < min_confidence:
                        continue
                    
                    # Process detection immediately to get crow ID
                    crow_id = tracker.process_detection(frame, frame_num, det, video_path, frame_time)
                    if crow_id:
                        detections_by_crow[crow_id].append((frame_num, det))
                        # Store minimal info for later crop extraction
                        frame_detection_info.append((frame_num, det, crow_id))
            
            # Clear frames from memory immediately
            frames.clear()
            frame_numbers.clear()
            
            # Force garbage collection every 10 batches
            if len(frame_detection_info) % (batch_size * 10) == 0:
                gc.collect()
            
            pbar.update(len(frames) if frames else batch_size)
    
    cap.release()
    
    # Filter for qualifying crows
    logger.info("Pass 2: Saving crops for qualifying crows...")
    qualifying_crows = {crow_id: dets for crow_id, dets in detections_by_crow.items() 
                       if len(dets) >= min_detections}
    
    crops_saved = 0
    total_qualifying_detections = sum(len(dets) for dets in qualifying_crows.values())
    
    # Group detections by frame number for efficient video access
    detections_by_frame = defaultdict(list)
    for frame_num, det, crow_id in frame_detection_info:
        if crow_id in qualifying_crows:
            detections_by_frame[frame_num].append((det, crow_id))
    
    # Sort frame numbers for sequential video access
    sorted_frame_nums = sorted(detections_by_frame.keys())
    
    logger.info(f"Extracting crops from {len(sorted_frame_nums)} frames for {len(qualifying_crows)} qualifying crows")
    
    # Process frames sequentially to minimize video seeking
    cap = cv2.VideoCapture(video_path)
    current_frame_num = -1
    
    with tqdm(total=len(sorted_frame_nums), desc="Extracting crops") as pbar:
        for target_frame_num in sorted_frame_nums:
            # Seek to frame if needed
            if current_frame_num != target_frame_num:
                cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_num)
                current_frame_num = target_frame_num
            
            ret, frame = cap.read()
            if not ret:
                logger.warning(f"Failed to read frame {target_frame_num}")
                continue
            
            # Process all detections for this frame
            for det, crow_id in detections_by_frame[target_frame_num]:
                # Limit to best detections per crow (memory optimization)
                crow_dets = qualifying_crows[crow_id]
                sorted_dets = sorted(crow_dets, key=lambda x: x[1]['score'], reverse=True)[:10]
                
                # Check if this detection is in the top 10 for this crow
                if (target_frame_num, det) in sorted_dets:
                    from tracking import extract_normalized_crow_crop
                    crop = extract_normalized_crow_crop(frame, det['bbox'], correct_orientation=True, padding=0.3)
                    if crop is not None:
                        crop_path = tracker.save_crop(crop, crow_id, target_frame_num, video_path)
                        if crop_path:
                            crops_saved += 1
            
            pbar.update(1)
    
    cap.release()
    
    # Log summary
    total_crows = len(detections_by_crow)
    crows_with_min_detections = len(qualifying_crows)
    logger.info(f"Found {total_crows} crows, {crows_with_min_detections} with {min_detections}+ detections")
    logger.info(f"Saved {crops_saved} crop files")
    
    return detections_by_crow

def find_video_files(video_dir, recursive=False):
    """
    Find video files in directory, optionally recursively.
    """
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.MP4', '.AVI', '.MOV', '.MKV')
    video_files = []
    
    if recursive:
        for root, dirs, files in os.walk(video_dir):
            for file in files:
                if file.endswith(video_extensions):
                    video_files.append(os.path.join(root, file))
    else:
        try:
            for file in os.listdir(video_dir):
                if file.endswith(video_extensions):
                    video_files.append(os.path.join(video_dir, file))
        except OSError as e:
            logger.error(f"Error reading directory {video_dir}: {e}")
            return []
    
    return sorted(video_files)

def main():
    parser = argparse.ArgumentParser(description="Memory-optimized extraction of crow crops from videos")
    parser.add_argument("video_dir", help="Directory containing input videos")
    parser.add_argument("--output-dir", default="crow_crops", help="Base directory to save crops")
    parser.add_argument("--min-confidence", type=float, default=0.2, help="Minimum detection confidence")
    parser.add_argument("--min-detections", type=int, default=2, help="Minimum detections per crow")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for processing (reduced for memory)")
    parser.add_argument("--multi-view-yolo", action="store_true", help="Enable multi-view for YOLO")
    parser.add_argument("--multi-view-rcnn", action="store_true", help="Enable multi-view for R-CNN")
    parser.add_argument("--nms-threshold", type=float, default=0.3, help="NMS/IOU threshold for merging boxes")
    parser.add_argument("--recursive", action="store_true", help="Search for videos recursively in subdirectories")
    parser.add_argument("--max-videos", type=int, help="Maximum number of videos to process (for testing)")
    
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
    logger.info(f"NMS threshold: {args.nms_threshold}, batch_size: {args.batch_size}")
    logger.info(f"Recursive search: {args.recursive}")
    
    # Find all video files
    video_files = find_video_files(args.video_dir, recursive=args.recursive)
    
    if not video_files:
        logger.warning(f"No video files found in {args.video_dir}")
        logger.info("Supported formats: .mp4, .avi, .mov, .mkv")
        sys.exit(0)
    
    # Limit videos if specified
    if args.max_videos:
        video_files = video_files[:args.max_videos]
        logger.info(f"Limited to first {args.max_videos} videos")
    
    logger.info(f"Found {len(video_files)} video files to process")
    if args.recursive:
        logger.info("Videos found in:")
        for video_file in video_files:
            rel_path = os.path.relpath(video_file, args.video_dir)
            logger.info(f"  {rel_path}")
    
    total_crows_before = len(tracker.list_crows())
    
    for i, video_file in enumerate(video_files, 1):
        rel_path = os.path.relpath(video_file, args.video_dir)
        logger.info(f"\nProcessing video {i}/{len(video_files)}: {rel_path}...")
        
        try:
            extract_crops_from_video_memory_optimized(
                video_file,
                tracker,
                min_confidence=args.min_confidence,
                min_detections=args.min_detections,
                batch_size=args.batch_size,
                multi_view_yolo=args.multi_view_yolo,
                multi_view_rcnn=args.multi_view_rcnn,
                nms_threshold=args.nms_threshold
            )
            # Force garbage collection between videos
            gc.collect()
            
        except Exception as e:
            logger.error(f"Error processing {rel_path}: {e}")
            continue
    
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