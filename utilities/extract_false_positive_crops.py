#!/usr/bin/env python3
"""
Extract low-confidence detections for manual review as potential "not_a_crow" examples.

This tool processes videos using the existing facebeak detection pipeline and extracts
crops from low-confidence detections that are likely false positives. These can then
be manually reviewed and labeled as hard negatives for training.
"""

import argparse
import os
import sys
import cv2
import numpy as np
from pathlib import Path
import logging
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detection import detect_crows_parallel
from crow_tracking import CrowTracker

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_false_positive_crops(
    video_path,
    output_dir="potential_not_crow_crops",
    min_confidence=0.2,
    max_confidence=0.6,
    frame_skip=15,
    max_crops_per_video=50,
    crop_size=(224, 224),
    batch_size=10
):
    """
    Extract crops from low-confidence detections for manual review.
    
    Args:
        video_path: Path to input video file
        output_dir: Directory to save extracted crops
        min_confidence: Minimum confidence to consider (lower = more false positives)
        max_confidence: Maximum confidence to consider (higher = more real crows)
        frame_skip: Skip every N frames for efficiency
        max_crops_per_video: Maximum number of crops to extract per video
        crop_size: Size to resize crops to (width, height)
        batch_size: Number of frames to process in each batch
    
    Returns:
        int: Number of crops extracted
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error(f"Could not open video: {video_path}")
        return 0
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    video_name = Path(video_path).stem
    
    logger.info(f"Processing video: {video_name}")
    logger.info(f"Looking for detections with confidence {min_confidence:.2f} - {max_confidence:.2f}")
    logger.info(f"Total frames: {total_frames}, FPS: {fps}")
    
    crops_extracted = 0
    frame_count = 0
    frames_batch = []
    frame_indices = []
    
    # Initialize tracker for crop extraction
    tracker = CrowTracker(enable_audio_extraction=False)
    
    # Process frames
    with tqdm(total=total_frames//frame_skip, desc="Extracting potential false positives") as pbar:
        while cap.isOpened() and crops_extracted < max_crops_per_video:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Skip frames for efficiency
            if frame_count % frame_skip != 0:
                frame_count += 1
                continue
            
            # Collect frames for batch processing
            frames_batch.append(frame.copy())
            frame_indices.append(frame_count)
            
            # Process batch when full
            if len(frames_batch) >= batch_size:
                extracted = process_frame_batch(
                    frames_batch, frame_indices, video_name, tracker,
                    output_path, min_confidence, max_confidence, crop_size,
                    crops_extracted, max_crops_per_video
                )
                crops_extracted += extracted
                frames_batch = []
                frame_indices = []
            
            frame_count += 1
            pbar.update(1)
            
            if crops_extracted >= max_crops_per_video:
                break
    
    # Process remaining frames
    if frames_batch and crops_extracted < max_crops_per_video:
        extracted = process_frame_batch(
            frames_batch, frame_indices, video_name, tracker,
            output_path, min_confidence, max_confidence, crop_size,
            crops_extracted, max_crops_per_video
        )
        crops_extracted += extracted
    
    cap.release()
    logger.info(f"Extracted {crops_extracted} potential false positive crops from {video_name}")
    return crops_extracted

def process_frame_batch(
    frames_batch, frame_indices, video_name, tracker,
    output_path, min_confidence, max_confidence, crop_size,
    current_count, max_crops
):
    """Process a batch of frames and extract crops from target confidence range."""
    try:
        # Use existing detection pipeline
        detections_batch = detect_crows_parallel(
            frames_batch, 
            score_threshold=min_confidence  # Use minimum threshold for detection
        )
        
        crops_extracted = 0
        
        for frame_idx, (frame, detections) in enumerate(zip(frames_batch, detections_batch)):
            if current_count + crops_extracted >= max_crops:
                break
                
            frame_num = frame_indices[frame_idx]
            
            for detection in detections:
                if current_count + crops_extracted >= max_crops:
                    break
                
                confidence = detection['score']
                
                # Only extract detections in our target confidence range
                if min_confidence <= confidence <= max_confidence:
                    bbox = detection['bbox']
                    
                    # Extract crop using the tracking module function
                    from tracking import extract_normalized_crow_crop
                    crop_result = extract_normalized_crow_crop(frame, bbox, expected_size=(224, 224), correct_orientation=True, padding=0.3)
                    if crop_result and 'full' in crop_result:
                        crop_image = crop_result['full']
                        
                        # Convert to uint8 and resize
                        if crop_image.dtype != np.uint8:
                            crop_image = (crop_image * 255).astype(np.uint8)
                        
                        crop_resized = cv2.resize(crop_image, crop_size, interpolation=cv2.INTER_LANCZOS4)
                        
                        # Save crop with descriptive filename
                        crop_filename = f"{video_name}_frame{frame_num:06d}_conf{confidence:.3f}_{current_count + crops_extracted:04d}.jpg"
                        crop_path = output_path / crop_filename
                        
                        success = cv2.imwrite(str(crop_path), crop_resized)
                        if success:
                            crops_extracted += 1
                            logger.debug(f"Extracted crop: {crop_filename} (confidence: {confidence:.3f})")
                        else:
                            logger.warning(f"Failed to save crop: {crop_path}")
        
        return crops_extracted
        
    except Exception as e:
        logger.warning(f"Error processing frame batch: {e}")
        return 0

def process_video_directory(
    input_dir, 
    output_dir="potential_not_crow_crops", 
    **kwargs
):
    """
    Process all videos in a directory to extract potential false positive crops.
    
    Args:
        input_dir: Directory containing video files
        output_dir: Directory to save extracted crops
        **kwargs: Additional arguments for extract_false_positive_crops
    
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
            crops = extract_false_positive_crops(video_file, output_dir, **kwargs)
            total_crops += crops
            
            # Log progress
            logger.info(f"Video {video_file.name}: {crops} crops extracted")
            
        except Exception as e:
            logger.error(f"Error processing {video_file}: {e}")
    
    logger.info(f"\n{'='*50}")
    logger.info(f"EXTRACTION COMPLETE")
    logger.info(f"{'='*50}")
    logger.info(f"Total potential false positive crops extracted: {total_crops}")
    logger.info(f"Saved to: {output_dir}")
    logger.info(f"\nNext steps:")
    logger.info(f"1. Review crops using: python gui/kivy_image_reviewer.py")
    logger.info(f"2. Label real crows as 'crow' and false positives as 'not_a_crow'")
    logger.info(f"3. Use labeled data for training hard negative examples")
    
    return total_crops

def validate_extraction_results(output_dir):
    """Validate extraction results and provide statistics."""
    output_path = Path(output_dir)
    if not output_path.exists():
        logger.warning(f"Output directory does not exist: {output_dir}")
        return
    
    # Count crops
    crop_files = list(output_path.glob("*.jpg"))
    total_crops = len(crop_files)
    
    if total_crops == 0:
        logger.warning("No crops found in output directory")
        return
    
    # Analyze confidence distribution from filenames
    confidences = []
    for crop_file in crop_files:
        try:
            # Extract confidence from filename: ..._conf0.456_...
            parts = crop_file.stem.split('_conf')
            if len(parts) > 1:
                conf_part = parts[1].split('_')[0]
                confidence = float(conf_part)
                confidences.append(confidence)
        except (ValueError, IndexError):
            pass
    
    logger.info(f"\nExtraction Results Summary:")
    logger.info(f"{'='*30}")
    logger.info(f"Total crops extracted: {total_crops}")
    
    if confidences:
        logger.info(f"Confidence range: {min(confidences):.3f} - {max(confidences):.3f}")
        logger.info(f"Average confidence: {np.mean(confidences):.3f}")
        
        # Confidence distribution
        low_conf = sum(1 for c in confidences if c < 0.4)
        med_conf = sum(1 for c in confidences if 0.4 <= c < 0.6)
        high_conf = sum(1 for c in confidences if c >= 0.6)
        
        logger.info(f"Distribution:")
        logger.info(f"  Low confidence (< 0.4): {low_conf}")
        logger.info(f"  Medium confidence (0.4-0.6): {med_conf}")
        logger.info(f"  High confidence (>= 0.6): {high_conf}")
    
    # Quick quality check
    logger.info(f"\nRecommendations:")
    if total_crops < 20:
        logger.info("- Consider lowering min_confidence or increasing max_crops_per_video")
    if confidences and max(confidences) > 0.7:
        logger.info("- Consider lowering max_confidence to focus on more likely false positives")
    
    logger.info(f"- Use image reviewer to label these crops")
    logger.info(f"- Focus on obvious false positives for 'not_a_crow' labels")

def main():
    parser = argparse.ArgumentParser(
        description="Extract low-confidence detections as potential false positives for manual review"
    )
    parser.add_argument("input", help="Input video file or directory")
    parser.add_argument("--output", "-o", default="potential_not_crow_crops", 
                       help="Output directory for crops (default: potential_not_crow_crops)")
    parser.add_argument("--min-confidence", type=float, default=0.2,
                       help="Minimum confidence threshold (default: 0.2)")
    parser.add_argument("--max-confidence", type=float, default=0.6,
                       help="Maximum confidence threshold (default: 0.6)")
    parser.add_argument("--skip", "-s", type=int, default=15,
                       help="Skip every N frames (default: 15)")
    parser.add_argument("--max-crops", "-m", type=int, default=50,
                       help="Maximum crops per video (default: 50)")
    parser.add_argument("--size", type=int, nargs=2, default=[224, 224],
                       help="Crop size as width height (default: 224 224)")
    parser.add_argument("--batch-size", type=int, default=10,
                       help="Batch size for processing (default: 10)")
    parser.add_argument("--validate", action="store_true",
                       help="Validate existing crops and show statistics")
    
    args = parser.parse_args()
    
    if args.validate:
        validate_extraction_results(args.output)
        return 0
    
    input_path = Path(args.input)
    crop_size = tuple(args.size)
    
    # Validate confidence range
    if args.min_confidence >= args.max_confidence:
        logger.error("min_confidence must be less than max_confidence")
        return 1
    
    if args.max_confidence > 0.8:
        logger.warning("max_confidence > 0.8 may include many real crows")
    
    if input_path.is_file():
        # Process single video
        crops = extract_false_positive_crops(
            input_path,
            args.output,
            args.min_confidence,
            args.max_confidence,
            args.skip,
            args.max_crops,
            crop_size,
            args.batch_size
        )
        print(f"Extracted {crops} potential false positive crops")
    elif input_path.is_dir():
        # Process directory
        crops = process_video_directory(
            input_path,
            args.output,
            min_confidence=args.min_confidence,
            max_confidence=args.max_confidence,
            frame_skip=args.skip,
            max_crops_per_video=args.max_crops,
            crop_size=crop_size,
            batch_size=args.batch_size
        )
    else:
        logger.error(f"Input path does not exist: {args.input}")
        return 1
    
    # Show validation results
    validate_extraction_results(args.output)
    
    return 0

if __name__ == "__main__":
    exit(main()) 