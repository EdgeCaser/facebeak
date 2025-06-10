#!/usr/bin/env python3
"""
Extract non-crow bird crops from video files for hard negative training.

This script processes video files to detect birds (excluding crows) and extracts
crop images that can be used as negative examples in training datasets.
Updated to integrate with the current database and pipeline architecture.
"""

import argparse
import os
import sys
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import logging
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db import add_image_label, get_connection
from detection import detect_crows_parallel, detect_crows_cascade

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_non_crow_crops(
    video_path,
    output_dir="non_crow_crops",
    confidence_threshold=0.5,
    frame_skip=10,
    max_crops_per_video=100,
    crop_size=(224, 224),  # Changed to match training pipeline
    use_existing_detection=True,
    detect_all_birds=False,
    auto_label=True
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
        use_existing_detection: Use the facebeak detection pipeline instead of raw YOLO
        detect_all_birds: If True, detect all birds; if False, only non-crow birds
        auto_label: Automatically add labels to database
    
    Returns:
        int: Number of crops extracted
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if not use_existing_detection:
        # Load YOLOv8 model for standalone operation
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
    
    # Collect frames for batch processing if using existing detection
    if use_existing_detection:
        frames_batch = []
        frame_indices = []
    
    # Process frames
    with tqdm(total=total_frames//frame_skip, desc="Extracting non-crow crops") as pbar:
        while cap.isOpened() and crops_extracted < max_crops_per_video:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Skip frames for efficiency
            if frame_count % frame_skip != 0:
                frame_count += 1
                continue
            
            if use_existing_detection:
                # Collect frames for batch processing
                frames_batch.append(frame.copy())
                frame_indices.append(frame_count)
                
                # Process in batches of 10 frames
                if len(frames_batch) >= 10:
                    crops_extracted += process_frame_batch_with_detection(
                        frames_batch, frame_indices, video_name, 
                        output_path, confidence_threshold, crop_size, 
                        detect_all_birds, auto_label, crops_extracted,
                        max_crops_per_video
                    )
                    frames_batch = []
                    frame_indices = []
            else:
                # Process individual frame with raw YOLO
                crops_extracted += process_frame_with_yolo(
                    frame, frame_count, video_name, model,
                    output_path, confidence_threshold, crop_size,
                    auto_label, crops_extracted, max_crops_per_video
                )
            
            frame_count += 1
            pbar.update(1)
            
            if crops_extracted >= max_crops_per_video:
                break
    
    # Process remaining frames if using batch detection
    if use_existing_detection and frames_batch:
        crops_extracted += process_frame_batch_with_detection(
            frames_batch, frame_indices, video_name,
            output_path, confidence_threshold, crop_size,
            detect_all_birds, auto_label, crops_extracted,
            max_crops_per_video
        )
    
    cap.release()
    logger.info(f"Extracted {crops_extracted} non-crow bird crops from {video_path}")
    return crops_extracted

def process_frame_batch_with_detection(
    frames_batch, frame_indices, video_name, output_path,
    confidence_threshold, crop_size, detect_all_birds,
    auto_label, current_count, max_crops
):
    """Process a batch of frames using the facebeak detection pipeline."""
    try:
        # Use the existing detection pipeline
        detections_batch = detect_crows_parallel(
            frames_batch, 
            score_threshold=confidence_threshold
        )
        
        crops_extracted = 0
        
        for frame_idx, (frame, detections) in enumerate(zip(frames_batch, detections_batch)):
            if current_count + crops_extracted >= max_crops:
                break
                
            frame_num = frame_indices[frame_idx]
            
            for detection in detections:
                if current_count + crops_extracted >= max_crops:
                    break
                
                # Filter based on detection strategy
                if not detect_all_birds:
                    # Only include detections that are likely non-crow birds
                    # Low-medium confidence detections from the crow detector
                    # might be other birds misclassified as crows
                    if detection['score'] > 0.7:  # Skip high-confidence crow detections
                        continue
                
                bbox = detection['bbox']
                crop_data = extract_and_save_crop(
                    frame, bbox, frame_num, video_name,
                    output_path, crop_size, current_count + crops_extracted
                )
                
                if crop_data:
                    crop_path, crop_filename = crop_data
                    
                    # Auto-label if enabled
                    if auto_label:
                        add_image_label(
                            crop_path, 
                            'not_a_crow', 
                            confidence=1.0 - detection['score'],  # Inverse confidence
                            reviewer_notes=f"Auto-labeled non-crow from {video_name}",
                            is_training_data=True  # Non-crow examples are good for training
                        )
                    
                    crops_extracted += 1
                    logger.debug(f"Extracted non-crow crop: {crop_filename}")
        
        return crops_extracted
        
    except Exception as e:
        logger.warning(f"Error processing frame batch: {e}")
        return 0

def process_frame_with_yolo(
    frame, frame_count, video_name, model, output_path,
    confidence_threshold, crop_size, auto_label, current_count, max_crops
):
    """Process a single frame using raw YOLO detection."""
    try:
        # Run YOLO detection
        results = model(frame, verbose=False)
        crops_extracted = 0
        
        # Process detections
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            
            for box in boxes:
                if current_count + crops_extracted >= max_crops:
                    break
                
                # Check if it's a bird (class 14 in COCO)
                class_id = int(box.cls.item())
                confidence = float(box.conf.item())
                
                if class_id == 14 and confidence >= confidence_threshold:  # Bird class
                    # Extract bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    bbox = [x1, y1, x2, y2]
                    
                    crop_data = extract_and_save_crop(
                        frame, bbox, frame_count, video_name,
                        output_path, crop_size, current_count + crops_extracted
                    )
                    
                    if crop_data:
                        crop_path, crop_filename = crop_data
                        
                        # Auto-label if enabled
                        if auto_label:
                            add_image_label(
                                crop_path,
                                'not_a_crow',
                                confidence=confidence,
                                reviewer_notes=f"Auto-labeled bird from YOLO detection in {video_name}",
                                is_training_data=True
                            )
                        
                        crops_extracted += 1
                        logger.debug(f"Extracted bird crop: {crop_filename}")
        
        return crops_extracted
        
    except Exception as e:
        logger.warning(f"Error processing frame {frame_count}: {e}")
        return 0

def extract_and_save_crop(frame, bbox, frame_num, video_name, output_path, crop_size, crop_index):
    """Extract and save a crop from the frame."""
    try:
        x1, y1, x2, y2 = map(int, bbox)
        
        # Validate bounding box
        if x2 <= x1 or y2 <= y1:
            return None
        
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
            return None
        
        # Resize crop to match training pipeline
        crop_resized = cv2.resize(crop, crop_size, interpolation=cv2.INTER_LANCZOS4)
        
        # Save crop
        crop_filename = f"{video_name}_frame{frame_num:06d}_noncrow{crop_index:04d}.jpg"
        crop_path = output_path / crop_filename
        
        success = cv2.imwrite(str(crop_path), crop_resized)
        if not success:
            logger.warning(f"Failed to save crop: {crop_path}")
            return None
        
        return str(crop_path), crop_filename
        
    except Exception as e:
        logger.warning(f"Error extracting crop: {e}")
        return None

def process_video_directory(
    input_dir, 
    output_dir="non_crow_crops", 
    **kwargs
):
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
    
    logger.info(f"Total non-crow crops extracted: {total_crops}")
    return total_crops

def validate_extracted_crops(output_dir):
    """Validate extracted crops and provide statistics."""
    output_path = Path(output_dir)
    if not output_path.exists():
        logger.warning(f"Output directory does not exist: {output_dir}")
        return
    
    # Count crops
    crop_files = list(output_path.glob("*.jpg"))
    total_crops = len(crop_files)
    
    # Check database labels
    labeled_count = 0
    unlabeled_count = 0
    
    for crop_file in crop_files:
        from db import get_image_label
        label_info = get_image_label(str(crop_file))
        if label_info:
            labeled_count += 1
        else:
            unlabeled_count += 1
    
    logger.info(f"Crop validation results:")
    logger.info(f"  Total crops: {total_crops}")
    logger.info(f"  Labeled in database: {labeled_count}")
    logger.info(f"  Unlabeled: {unlabeled_count}")
    
    # Check for potential black images
    black_count = 0
    for crop_file in crop_files[:min(100, len(crop_files))]:  # Sample first 100
        try:
            img = cv2.imread(str(crop_file))
            if img is not None:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                mean_val = np.mean(gray)
                if mean_val < 10:  # Very dark
                    black_count += 1
        except Exception:
            pass
    
    if black_count > 0:
        logger.warning(f"Found {black_count} potentially black crops in sample")

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
    parser.add_argument("--size", type=int, nargs=2, default=[224, 224],
                       help="Crop size as width height (default: 224 224)")
    parser.add_argument("--use-yolo", action="store_true",
                       help="Use raw YOLO detection instead of facebeak pipeline")
    parser.add_argument("--detect-all-birds", action="store_true",
                       help="Detect all birds, not just potential non-crows")
    parser.add_argument("--no-auto-label", action="store_true",
                       help="Don't automatically add labels to database")
    parser.add_argument("--validate", action="store_true",
                       help="Validate extracted crops and show statistics")
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    crop_size = tuple(args.size)
    
    if args.validate:
        validate_extracted_crops(args.output)
        return 0
    
    if input_path.is_file():
        # Process single video
        crops = extract_non_crow_crops(
            input_path,
            args.output,
            args.confidence,
            args.skip,
            args.max_crops,
            crop_size,
            use_existing_detection=not args.use_yolo,
            detect_all_birds=args.detect_all_birds,
            auto_label=not args.no_auto_label
        )
        print(f"Extracted {crops} non-crow bird crops")
    elif input_path.is_dir():
        # Process directory
        crops = process_video_directory(
            input_path,
            args.output,
            confidence_threshold=args.confidence,
            frame_skip=args.skip,
            max_crops_per_video=args.max_crops,
            crop_size=crop_size,
            use_existing_detection=not args.use_yolo,
            detect_all_birds=args.detect_all_birds,
            auto_label=not args.no_auto_label
        )
        print(f"Total extracted {crops} non-crow bird crops")
    else:
        logger.error(f"Input path does not exist: {args.input}")
        return 1
    
    # Validate results
    if not args.validate:
        validate_extracted_crops(args.output)
    
    return 0

if __name__ == "__main__":
    exit(main()) 