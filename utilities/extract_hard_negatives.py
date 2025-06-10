#!/usr/bin/env python3
"""
Extract hard negative crops (non-bird objects) from videos for training data.
This tool samples random crops from videos while avoiding detected bird areas.
"""

import os
import sys
import cv2
import numpy as np
import sqlite3
import argparse
import random
from pathlib import Path
from typing import List, Tuple, Set
from tqdm import tqdm
import logging

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from detection import detect_crows_parallel
from db import get_connection, get_image_label, add_image_label
from tracking import CROW_EMBEDDING_MODEL
from tracking import extract_normalized_crow_crop
import torch
import torchvision.transforms as transforms

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_bird_areas(detections: List[dict]) -> List[Tuple[int, int, int, int]]:
    """Extract bird bounding boxes from detections."""
    bird_areas = []
    for detection in detections:
        if detection.get('class') == 'bird':
            bbox = detection['bbox']
            # Convert to integers and add some padding
            x1 = max(0, int(bbox[0]) - 20)
            y1 = max(0, int(bbox[1]) - 20)
            x2 = int(bbox[2]) + 20
            y2 = int(bbox[3]) + 20
            bird_areas.append((x1, y1, x2, y2))
    return bird_areas

def is_overlapping(crop_box: Tuple[int, int, int, int], bird_areas: List[Tuple[int, int, int, int]], 
                  min_overlap_threshold: float = 0.1) -> bool:
    """Check if crop overlaps significantly with any bird area."""
    cx1, cy1, cx2, cy2 = crop_box
    crop_area = (cx2 - cx1) * (cy2 - cy1)
    
    for bx1, by1, bx2, by2 in bird_areas:
        # Calculate intersection
        ix1 = max(cx1, bx1)
        iy1 = max(cy1, by1)
        ix2 = min(cx2, bx2)
        iy2 = min(cy2, by2)
        
        if ix1 < ix2 and iy1 < iy2:
            intersection_area = (ix2 - ix1) * (iy2 - iy1)
            overlap_ratio = intersection_area / crop_area
            if overlap_ratio > min_overlap_threshold:
                return True
    
    return False

def generate_random_crops(frame_width: int, frame_height: int, 
                         bird_areas: List[Tuple[int, int, int, int]], 
                         num_attempts: int = 100) -> List[Tuple[int, int, int, int]]:
    """Generate random crop locations that don't overlap significantly with bird areas."""
    valid_crops = []
    crop_size = 512  # Fixed size to match existing pipeline
    
    for _ in range(num_attempts):
        # Generate random top-left corner
        max_x = frame_width - crop_size
        max_y = frame_height - crop_size
        
        if max_x <= 0 or max_y <= 0:
            continue
            
        x = random.randint(0, max_x)
        y = random.randint(0, max_y)
        
        crop_box = (x, y, x + crop_size, y + crop_size)
        
        # Check if this crop overlaps with bird areas
        if not is_overlapping(crop_box, bird_areas):
            valid_crops.append(crop_box)
    
    return valid_crops

def extract_crop_from_frame(frame: np.ndarray, crop_box: Tuple[int, int, int, int]) -> np.ndarray:
    """Extract crop from frame (already 512x512)."""
    x1, y1, x2, y2 = crop_box
    crop = frame[y1:y2, x1:x2]
    
    # No resizing needed - crops are already 512x512 to match pipeline
    return crop

def is_crop_interesting(crop: np.ndarray, min_std: float = 10.0) -> bool:
    """Check if crop has enough visual variation to be interesting for training."""
    # Convert to grayscale and check standard deviation
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    std_dev = np.std(gray)
    
    # Also check if it's not mostly one color (like sky or water)
    mean_val = np.mean(gray)
    
    return std_dev > min_std and not (mean_val < 30 or mean_val > 200)

def process_video_file(video_path: str, output_dir: str, max_crops_per_video: int = 200, 
                      frame_skip: int = 30) -> int:
    """Process a single video file and extract hard negative crops."""
    logger.info(f"Processing video: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Could not open video: {video_path}")
        return 0
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0
    
    logger.info(f"Video stats: {total_frames} frames, {fps:.2f} fps, {duration:.2f} seconds")
    
    video_name = Path(video_path).stem
    video_output_dir = Path(output_dir) / video_name
    video_output_dir.mkdir(parents=True, exist_ok=True)
    
    extracted_count = 0
    
    # Sample frames throughout the video
    frame_indices = list(range(0, total_frames, frame_skip))
    
    with tqdm(total=len(frame_indices), desc=f"Processing {video_name}") as pbar:
        for target_frame in frame_indices:
            if extracted_count >= max_crops_per_video:
                logger.info(f"Reached maximum crops limit ({max_crops_per_video}) for {video_name}")
                break
                
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
            ret, frame = cap.read()
            
            if not ret:
                pbar.update(1)
                continue
            
            frame_height, frame_width = frame.shape[:2]
            
            # Run detection on this frame
            try:
                detections_list = detect_crows_parallel([frame])
                detections = detections_list[0] if detections_list else []
            except Exception as e:
                logger.warning(f"Detection failed for frame {target_frame}: {e}")
                detections = []
            
            # Get bird areas to avoid
            bird_areas = get_bird_areas(detections)
            
            # Generate random crops that don't overlap with birds
            potential_crops = generate_random_crops(frame_width, frame_height, bird_areas)
            
            # Extract and save ALL interesting crops from this frame
            for i, crop_box in enumerate(potential_crops):
                if extracted_count >= max_crops_per_video:
                    break
                    
                crop = extract_crop_from_frame(frame, crop_box)
                
                # Check if crop is visually interesting
                if is_crop_interesting(crop):
                    # Save crop
                    crop_filename = f"{video_name}_frame{target_frame:06d}_crop{i:02d}.jpg"
                    crop_path = video_output_dir / crop_filename
                    
                    cv2.imwrite(str(crop_path), crop)
                    extracted_count += 1
                    
                    logger.debug(f"Extracted crop: {crop_filename}")
            
            pbar.update(1)
    
    cap.release()
    logger.info(f"Extracted {extracted_count} hard negative crops from {video_name}")
    return extracted_count

def generate_embedding_from_image(image_path: str) -> np.ndarray:
    """Generate embedding from image file using the global embedding model."""
    if CROW_EMBEDDING_MODEL is None:
        logger.error("CROW_EMBEDDING_MODEL is not available")
        return None
    
    try:
        # Load and preprocess image
        import cv2
        from PIL import Image
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Could not load image: {image_path}")
            return None
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL and apply transforms
        pil_image = Image.fromarray(image)
        
        # Apply standard transforms (512x512 to match pipeline)
        transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Transform and add batch dimension
        img_tensor = transform(pil_image).unsqueeze(0)
        
        # Move to device
        device = next(CROW_EMBEDDING_MODEL.parameters()).device
        img_tensor = img_tensor.to(device)
        
        # Generate embedding
        with torch.no_grad():
            CROW_EMBEDDING_MODEL.eval()
            embedding = CROW_EMBEDDING_MODEL(img_tensor)
            embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
            embedding_np = embedding.squeeze().cpu().numpy()
        
        return embedding_np
        
    except Exception as e:
        logger.error(f"Error generating embedding for {image_path}: {e}")
        return None

def add_crops_to_database(crops_dir: str, label: str = "not_a_crow", confidence: float = 0.9) -> int:
    """Add extracted crops to database with labels."""
    logger.info(f"Adding crops from {crops_dir} to database with label '{label}', confidence={confidence}")
    
    added_count = 0
    crop_files = list(Path(crops_dir).rglob("*.jpg"))
    
    with tqdm(total=len(crop_files), desc="Adding to database") as pbar:
        for crop_path in crop_files:
            try:
                # Check if already in database
                existing_label = get_image_label(str(crop_path))
                if existing_label is not None:
                    logger.debug(f"Skipping {crop_path.name}, already in database")
                    pbar.update(1)
                    continue
                
                # Add to database with label and confidence
                add_image_label(str(crop_path), label, confidence=confidence,
                              reviewer_notes=f"Auto-labeled hard negative crop",
                              is_training_data=True)
                added_count += 1
                
            except Exception as e:
                logger.error(f"Error processing {crop_path}: {e}")
            
            pbar.update(1)
    
    logger.info(f"Added {added_count} new hard negative crops to database")
    return added_count

def main():
    parser = argparse.ArgumentParser(description="Extract hard negative crops from videos")
    parser.add_argument("input_path", help="Path to video file or directory containing videos")
    parser.add_argument("--output-dir", "-o", default="crow_crops/hard_negatives", 
                       help="Output directory for extracted crops")
    parser.add_argument("--max-crops-per-video", "-c", type=int, default=200,
                       help="Maximum number of crops to extract per video")
    parser.add_argument("--frame-skip", "-s", type=int, default=30,
                       help="Number of frames to skip between samples")
    parser.add_argument("--add-to-database", "-d", action="store_true",
                       help="Add extracted crops to database with 'not_a_crow' label")
    parser.add_argument("--label", default="not_a_crow",
                       help="Label to use when adding to database (must be: crow, not_a_crow, bad_crow, not_sure, multi_crow)")
    parser.add_argument("--confidence", type=float, default=0.9,
                       help="Confidence value when adding to database (0.1 to 1.0)")
    parser.add_argument("--video-extensions", nargs="+", 
                       default=[".mp4", ".avi", ".mov", ".mkv", ".MOV"],
                       help="Video file extensions to process")
    
    args = parser.parse_args()
    
    # Validate arguments
    valid_labels = ['crow', 'not_a_crow', 'bad_crow', 'not_sure', 'multi_crow']
    if args.label not in valid_labels:
        logger.error(f"Invalid label '{args.label}'. Must be one of: {', '.join(valid_labels)}")
        return
        
    if not 0.1 <= args.confidence <= 1.0:
        logger.error(f"Invalid confidence {args.confidence}. Must be between 0.1 and 1.0")
        return
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    input_path = Path(args.input_path)
    total_extracted = 0
    
    if input_path.is_file():
        # Process single video
        if input_path.suffix.lower() in [ext.lower() for ext in args.video_extensions]:
            total_extracted = process_video_file(
                str(input_path), str(output_dir), 
                args.max_crops_per_video, args.frame_skip
            )
        else:
            logger.error(f"File {input_path} is not a supported video format")
            return
    
    elif input_path.is_dir():
        # Process all videos in directory
        video_files = []
        for ext in args.video_extensions:
            video_files.extend(input_path.rglob(f"*{ext}"))
            video_files.extend(input_path.rglob(f"*{ext.upper()}"))
        
        logger.info(f"Found {len(video_files)} video files to process")
        
        for video_file in video_files:
            extracted = process_video_file(
                str(video_file), str(output_dir),
                args.max_crops_per_video, args.frame_skip
            )
            total_extracted += extracted
    
    else:
        logger.error(f"Input path {input_path} does not exist")
        return
    
    logger.info(f"Total extracted crops: {total_extracted}")
    
    # Add to database if requested
    if args.add_to_database and total_extracted > 0:
        added_count = add_crops_to_database(str(output_dir), args.label, args.confidence)
        logger.info(f"Database updated with {added_count} new entries")

if __name__ == "__main__":
    main() 