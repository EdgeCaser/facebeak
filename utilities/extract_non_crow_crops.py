#!/usr/bin/env python3
"""
Extract non-crow bird crops from video files for hard negative training.

This script processes video files to detect birds (excluding crows) and extracts
crop images that can be used as negative examples in training datasets.
"""

import argparse
import os
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import logging
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_non_crow_crops(
    video_path,
    output_dir="non_crow_crops",
    confidence_threshold=0.5,
    frame_skip=10,
    max_crops_per_video=100,
    crop_size=(512, 512)
):
    """
    Extract crops of non-crow birds from a video file.
    
    Args:
        video_path: Path to input video file
        output_dir: Directory to save extracted crops
        confidence_threshold: Minimum confidence for bird detection
        frame_skip: Skip every N frames for efficiency
        max_crops_per_video: Maximum number of crops to extract per video
        crop_size: Size to resize crops to (width, height)
    
    Returns:
        int: Number of crops extracted
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load YOLOv8 model
    try:
        model = YOLO('yolov8s.pt')
        logger.info("Loaded YOLOv8 model")
    except Exception as e:
        logger.error(f"Failed to load YOLOv8 model: {e}")
        return 0
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error(f"Could not open video: {video_path}")
        return 0
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    logger.info(f"Processing video: {video_path}")
    logger.info(f"Total frames: {total_frames}, FPS: {fps}")
    
    crops_extracted = 0
    frame_count = 0
    video_name = Path(video_path).stem
    
    # Process frames
    with tqdm(total=total_frames//frame_skip, desc="Extracting crops") as pbar:
        while cap.isOpened() and crops_extracted < max_crops_per_video:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Skip frames for efficiency
            if frame_count % frame_skip != 0:
                frame_count += 1
                continue
            
            try:
                # Run YOLO detection
                results = model(frame, verbose=False)
                
                # Process detections
                for result in results:
                    boxes = result.boxes
                    if boxes is None:
                        continue
                    
                    for box in boxes:
                        # Check if it's a bird (class 14 in COCO)
                        class_id = int(box.cls.item())
                        confidence = float(box.conf.item())
                        
                        if class_id == 14 and confidence >= confidence_threshold:  # Bird class
                            # Extract bounding box coordinates
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                            
                            # Validate bounding box
                            if x2 <= x1 or y2 <= y1:
                                continue
                            
                            # Add padding around detection
                            padding = 0.1
                            h, w = frame.shape[:2]
                            pad_w = int((x2 - x1) * padding)
                            pad_h = int((y2 - y1) * padding)
                            
                            x1 = max(0, x1 - pad_w)
                            y1 = max(0, y1 - pad_h)
                            x2 = min(w, x2 + pad_w)
                            y2 = min(h, y2 + pad_h)
                            
                            # Extract crop
                            crop = frame[y1:y2, x1:x2]
                            
                            # Skip very small crops
                            if crop.shape[0] < 32 or crop.shape[1] < 32:
                                continue
                            
                            # Resize crop
                            crop_resized = cv2.resize(crop, crop_size, interpolation=cv2.INTER_LANCZOS4)
                            
                            # Save crop
                            crop_filename = f"{video_name}_frame{frame_count}_bird{crops_extracted:04d}.jpg"
                            crop_path = output_path / crop_filename
                            
                            cv2.imwrite(str(crop_path), crop_resized)
                            crops_extracted += 1
                            
                            # Check if we've reached the limit
                            if crops_extracted >= max_crops_per_video:
                                break
                    
                    if crops_extracted >= max_crops_per_video:
                        break
                        
            except Exception as e:
                logger.warning(f"Error processing frame {frame_count}: {e}")
            
            frame_count += 1
            pbar.update(1)
    
    cap.release()
    logger.info(f"Extracted {crops_extracted} bird crops from {video_path}")
    return crops_extracted

def process_video_directory(input_dir, output_dir="non_crow_crops", **kwargs):
    """
    Process all videos in a directory to extract non-crow bird crops.
    
    Args:
        input_dir: Directory containing video files
        output_dir: Directory to save extracted crops
        **kwargs: Additional arguments for extract_non_crow_crops
    
    Returns:
        int: Total number of crops extracted
    """
    input_path = Path(input_dir)
    if not input_path.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        return 0
    
    # Find video files
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'}
    video_files = [f for f in input_path.iterdir() 
                   if f.suffix.lower() in video_extensions]
    
    if not video_files:
        logger.warning(f"No video files found in {input_dir}")
        return 0
    
    logger.info(f"Found {len(video_files)} video files")
    
    total_crops = 0
    for video_file in video_files:
        try:
            crops = extract_non_crow_crops(video_file, output_dir, **kwargs)
            total_crops += crops
        except Exception as e:
            logger.error(f"Error processing {video_file}: {e}")
    
    logger.info(f"Total crops extracted: {total_crops}")
    return total_crops

def main():
    parser = argparse.ArgumentParser(description="Extract non-crow bird crops from videos")
    parser.add_argument("input", help="Input video file or directory")
    parser.add_argument("--output", "-o", default="non_crow_crops", 
                       help="Output directory for crops (default: non_crow_crops)")
    parser.add_argument("--confidence", "-c", type=float, default=0.5,
                       help="Minimum confidence threshold (default: 0.5)")
    parser.add_argument("--skip", "-s", type=int, default=10,
                       help="Skip every N frames (default: 10)")
    parser.add_argument("--max-crops", "-m", type=int, default=100,
                       help="Maximum crops per video (default: 100)")
    parser.add_argument("--size", type=int, nargs=2, default=[512, 512],
                       help="Crop size as width height (default: 512 512)")
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    crop_size = tuple(args.size)
    
    if input_path.is_file():
        # Process single video
        crops = extract_non_crow_crops(
            input_path,
            args.output,
            args.confidence,
            args.skip,
            args.max_crops,
            crop_size
        )
        print(f"Extracted {crops} bird crops")
    elif input_path.is_dir():
        # Process directory
        crops = process_video_directory(
            input_path,
            args.output,
            confidence_threshold=args.confidence,
            frame_skip=args.skip,
            max_crops_per_video=args.max_crops,
            crop_size=crop_size
        )
        print(f"Total extracted {crops} bird crops")
    else:
        logger.error(f"Input path does not exist: {args.input}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 