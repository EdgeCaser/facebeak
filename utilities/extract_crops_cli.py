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

# Configure logging
logger = setup_logging()

def extract_crops_from_video(video_path, tracker, min_confidence=0.2, min_detections=2, 
                           batch_size=32, multi_view_yolo=False, multi_view_rcnn=False, 
                           nms_threshold=0.3):
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
    
    frames = []
    frame_numbers = []
    detections_by_crow = defaultdict(list)
    
    logger.info("Extracting frames and detecting crows...")
    with tqdm(total=total_frames) as pbar:
        while True:
            frames = []
            frame_numbers = []
            for _ in range(batch_size):
                ret, frame = cap.read()
                if not ret:
                    break
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
            
            logger.info(f"Batch of {len(frames)} frames: found {sum(len(d) for d in detections)} total detections")
            
            for frame_idx, frame_dets in enumerate(detections):
                frame = frames[frame_idx]
                frame_num = frame_numbers[frame_idx]
                frame_time = frame_num / fps if fps > 0 else None
                
                if frame_dets:
                    logger.info(f"Frame {frame_num}: Found {len(frame_dets)} detections")
                    for det in frame_dets:
                        logger.info(f"  Detection: class={det.get('class', 'unknown')}, score={det['score']:.3f}, bbox={det['bbox']}")
                
                for det in frame_dets:
                    if det['score'] < min_confidence:
                        continue
                    
                    # Process detection and get crow_id
                    crow_id = tracker.process_detection(frame, frame_num, det, video_path, frame_time)
                    if crow_id:
                        detections_by_crow[crow_id].append((frame_num, det))
            
            pbar.update(len(frames))
    
    cap.release()
    
    # Log summary
    total_crows = len(detections_by_crow)
    crows_with_min_detections = sum(1 for dets in detections_by_crow.values() if len(dets) >= min_detections)
    logger.info(f"Found {total_crows} crows, {crows_with_min_detections} with {min_detections}+ detections")
    
    return detections_by_crow

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
    
    args = parser.parse_args()
    
    # Initialize crow tracker
    tracker = CrowTracker(args.output_dir)
    
    # Create processing run directory
    run_dir = tracker.create_processing_run()
    logger.info(f"Starting extraction with run directory: {run_dir}")
    logger.info(f"Parameters: confidence={args.min_confidence}, min_detections={args.min_detections}")
    logger.info(f"Multi-view: YOLO={args.multi_view_yolo}, RCNN={args.multi_view_rcnn}")
    logger.info(f"NMS threshold: {args.nms_threshold}")
    
    # Process all videos in directory
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv')
    video_files = [f for f in os.listdir(args.video_dir) if f.lower().endswith(video_extensions)]
    logger.info(f"Found {len(video_files)} video files to process")
    
    total_crows_before = len(tracker.list_crows())
    
    for video_file in video_files:
        video_path = os.path.join(args.video_dir, video_file)
        logger.info(f"\nProcessing {video_file}...")
        extract_crops_from_video(
            video_path,
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

if __name__ == "__main__":
    main() 