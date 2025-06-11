import os
import sys
import cv2
import numpy as np
from tqdm import tqdm
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from detection import detect_crows_parallel
from tracking import extract_normalized_crow_crop
import torch
from pathlib import Path
import shutil
from datetime import datetime
from crow_tracking import CrowTracker
from collections import defaultdict
from logging_config import setup_logging

import decord # Added for decord
decord.bridge.set_bridge('torch') # Or 'numpy' if preferred, then convert tensor to numpy

# Configure logging
logger = setup_logging()

def extract_crops_from_video_task(
    video_path_str: str,
    base_dir_str: str,
    min_confidence_val: float,
    frame_skip_val: int,
    batch_size_frames_val: int, # Batch size for reading/detecting frames
    enable_audio_val: bool,
    correct_orientation_val: bool,
    target_frame_rate_val: int = None
):
    """
    Extracts crow crops from a single video for training.
    Instantiates its own CrowTracker.
    """
    logger.info(f"Task started for video: {video_path_str}")
    logger.info(f"  Base directory: {base_dir_str}")
    logger.info(f"  Min confidence: {min_confidence_val}")
    logger.info(f"  Frame skip: {frame_skip_val}")
    logger.info(f"  Frame batch size: {batch_size_frames_val}")
    logger.info(f"  Target FPS: {target_frame_rate_val if target_frame_rate_val else 'Not set'}")

    try:
        # NEW: Output to dataset/crows/generic/<video_name> for generic crows
        # For future: use dataset/crows/<crow_id>/<video_name> for identified crows
        video_name = Path(video_path_str).stem
        base_dir_str = os.path.join('dataset', 'crows', 'generic', video_name)
        os.makedirs(base_dir_str, exist_ok=True)
        tracker = CrowTracker(
            base_dir=base_dir_str,
            enable_audio_extraction=enable_audio_val,
            correct_orientation=correct_orientation_val
        )
    except Exception as e:
        logger.error(f"Failed to initialize CrowTracker for video {video_path_str}: {e}", exc_info=True)
        return {
            "video_path": video_path_str, "status": "failure", "crops_extracted": 0,
            "detections_count": 0, "crows_identified_count": 0,
            "error_message": f"CrowTracker initialization failed: {e}"
        }

    video_reader = None
    use_decord = True # Flag to control which reader is used
    try:
        # Attempt to use decord
        video_reader = decord.VideoReader(video_path_str, ctx=decord.cpu(0))
        logger.info(f"Successfully opened video with decord: {video_path_str}")
        total_frames_video = len(video_reader)
        video_fps = video_reader.get_avg_fps()
    except Exception as e_decord: # Catch generic decord errors, could be decord.DECORDError too
        logger.warning(f"Failed to open video {video_path_str} with decord: {e_decord}. Falling back to OpenCV.")
        use_decord = False
        video_reader = cv2.VideoCapture(video_path_str)
        if not video_reader.isOpened():
            logger.error(f"Failed to open video with OpenCV after decord failed: {video_path_str}")
            return {
                "video_path": video_path_str, "status": "failure", "crops_extracted": 0,
                "detections_count": 0, "crows_identified_count": 0,
                "error_message": "Failed to open video with both decord and OpenCV"
            }
        total_frames_video = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = video_reader.get(cv2.CAP_PROP_FPS)

    logger.info(f"Video info for {Path(video_path_str).name}: {total_frames_video} frames, {video_fps:.2f} FPS (using {'decord' if use_decord else 'OpenCV'})")

    actual_frame_skip = frame_skip_val
    if target_frame_rate_val and video_fps > 0 and target_frame_rate_val < video_fps:
        actual_frame_skip = max(0, round(video_fps / target_frame_rate_val) -1) # if skip is 0, process every frame at target rate logic
        logger.info(f"Adjusted frame skip to {actual_frame_skip} for target FPS {target_frame_rate_val} (original FPS: {video_fps:.2f})")
    
    processed_frames_count = 0
    total_detections_in_video = 0
    video_specific_crow_ids = set()
    run_dir = tracker.create_processing_run()

    with tqdm(total=total_frames_video, desc=f"Processing {Path(video_path_str).name}") as pbar:
        frame_num_iterator = 0
        while frame_num_iterator < total_frames_video:
            frames_to_fetch_indices = []
            current_batch_target_size = 0

            temp_frame_num = frame_num_iterator
            while current_batch_target_size < batch_size_frames_val and temp_frame_num < total_frames_video:
                frames_to_fetch_indices.append(temp_frame_num)
                current_batch_target_size += 1
                temp_frame_num += (actual_frame_skip + 1)

            if not frames_to_fetch_indices:
                break

            frames_batch_np = []
            if use_decord:
                try:
                    # Decord gives RGB tensors. Convert to NumPy BGR for detect_crows_parallel
                    frames_torch_batch = video_reader.get_batch(frames_to_fetch_indices)
                    frames_batch_np = [(cv2.cvtColor(frame.numpy(), cv2.COLOR_RGB2BGR)) for frame in frames_torch_batch]
                except Exception as e_decord_batch:
                    logger.error(f"Error fetching batch with decord from {video_path_str} (indices: {frames_to_fetch_indices}): {e_decord_batch}. Skipping batch.")
                    # Advance iterator past the problematic part to avoid infinite loop
                    frame_num_iterator = frames_to_fetch_indices[-1] + actual_frame_skip + 1
                    pbar.update(len(frames_to_fetch_indices) * (actual_frame_skip + 1)) # Approximate update
                    continue
            else: # OpenCV
                for frame_idx_to_fetch in frames_to_fetch_indices:
                    video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_idx_to_fetch)
                    ret, frame = video_reader.read()
                    if not ret:
                        break
                    frames_batch_np.append(frame)
            
            frame_numbers_batch = frames_to_fetch_indices[:len(frames_batch_np)] # Ensure frame_numbers matches actual frames read

            if not frames_batch_np:
                break

            # Update progress bar based on the last frame index attempted in this batch
            pbar.update(frames_to_fetch_indices[-1] - pbar.n +1 if frames_to_fetch_indices else 0)
            frame_num_iterator = frames_to_fetch_indices[-1] + actual_frame_skip + 1


            detections_batch = detect_crows_parallel(frames_batch_np, score_threshold=min_confidence_val)
            
            for i, frame_dets in enumerate(detections_batch):
                current_frame = frames_batch_np[i]
                current_frame_num = frame_numbers_batch[i] # This should be correct from frames_to_fetch_indices
                current_frame_time = current_frame_num / video_fps if video_fps > 0 else None
                
                num_dets_in_frame = len(frame_dets)
                total_detections_in_video += num_dets_in_frame
                # processed_frames_count is implicitly len(frames_batch_np) for this batch,
                # but a running total processed_frames_count is more about distinct frames passed to detection
                # This definition might need review based on desired output.
                # Let's consider processed_frames_count as frames on which detection was run.
                processed_frames_count += 1


                if num_dets_in_frame > 0:
                    logger.debug(f"Video {Path(video_path_str).name}, Frame {current_frame_num}: Found {num_dets_in_frame} detections")
                
                for det in frame_dets:
                    if det['score'] < min_confidence_val:
                        continue
                    
                    crow_id = tracker.process_detection(
                        current_frame, current_frame_num, det,
                        video_path_str, current_frame_time
                    )
                    if crow_id:
                        video_specific_crow_ids.add(crow_id)

    if use_decord:
        # Decord VideoReader does not have a release() method explicitly. It's closed when the object is deleted.
        del video_reader
    else:
        video_reader.release() # OpenCV's VideoCapture

    tracker.cleanup_processing_dir(run_dir)

    logger.info(f"Finished video: {video_path_str}. Processed approximately {processed_frames_count} frames (considering batches). "
                f"Total detections: {total_detections_in_video}. Identified unique crow IDs in video: {len(video_specific_crow_ids)}.")
    
    return {
        "video_path": video_path_str,
        "status": "success",
        "frames_processed_in_video": processed_frames_count,
        "detections_count_in_video": total_detections_in_video,
        "crows_identified_count_in_video": len(video_specific_crow_ids),
        "error_message": None
    }

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Extract crow crops from videos for training")
    parser.add_argument("video_dir", help="Directory containing input videos")
    parser.add_argument("--output-dir", default="crow_crops", help="Base directory to save crops and metadata")
    parser.add_argument("--min-confidence", type=float, default=0.2, help="Minimum detection confidence")
    # parser.add_argument("--min-detections", type=int, default=2, help="Minimum detections per crow (Tracker internal logic)") # This is now handled by tracker logic
    parser.add_argument("--frame-skip", type=int, default=0, help="Number of frames to skip between processing (0 means process every frame). Overridden by --target-fps if set.")
    parser.add_argument("--target-fps", type=int, default=None, help="Target FPS for processing. If set, overrides --frame-skip.")
    parser.add_argument("--frame-batch-size", type=int, default=16, help="Batch size of frames for detection (default: 16, adjust based on GPU memory)")
    parser.add_argument("--enable-audio", action='store_true', help="Enable audio segment extraction by CrowTracker.")
    parser.add_argument("--correct-orientation", action='store_true', help="Enable automatic orientation correction by CrowTracker.")
    parser.add_argument("--skip-processed", action='store_true', help="Skip videos if a tracking data JSON already exists (Not implemented in this refactor).")


    args = parser.parse_args()
    
    # Adjust batch size based on available GPU memory if CUDA is available (heuristic)
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)  # GB
        if args.frame_batch_size > 8 : # Only adjust if user hasn't set a low value
            if gpu_memory < 4:
                args.frame_batch_size = max(4, args.frame_batch_size // 2)
                logger.info(f"Reduced frame_batch_size to {args.frame_batch_size} due to low GPU memory ({gpu_memory:.1f} GB)")
            elif gpu_memory > 10 and args.frame_batch_size < 64 : # Example: increase if >10GB and current batch < 64
                args.frame_batch_size = min(32, args.frame_batch_size * 2) # Cap increase
                logger.info(f"Increased frame_batch_size to {args.frame_batch_size} due to high GPU memory ({gpu_memory:.1f} GB)")
    
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.MP4', '.AVI', '.MOV', '.MKV') # Added uppercase
    video_files_found = []
    for root, _, files in os.walk(args.video_dir):
        for f in files:
            if f.lower().endswith(video_extensions):
                video_files_found.append(os.path.join(root, f))
    
    logger.info(f"Found {len(video_files_found)} video files to process in {args.video_dir} (recursive).")
    video_files_found.sort() # Process in a consistent order

    all_summaries = []

    # Sequential processing using the new task function
    for video_path_str in video_files_found:
        logger.info(f"\nProcessing video: {video_path_str}...")
        summary = extract_crops_from_video_task(
            video_path_str=video_path_str,
            base_dir_str=args.output_dir,
            min_confidence_val=args.min_confidence,
            frame_skip_val=args.frame_skip,
            batch_size_frames_val=args.frame_batch_size,
            enable_audio_val=args.enable_audio,
            correct_orientation_val=args.correct_orientation,
            target_frame_rate_val=args.target_fps
        )
        logger.info(f"Summary for {video_path_str}: {summary}")
        all_summaries.append(summary)
    
    # TODO (Future): Implement multiprocessing for parallel video processing
    # Example structure:
    # from functools import partial
    # from multiprocessing import Pool, cpu_count
    #
    # # Determine number of workers (e.g., number of GPUs or CPU cores)
    # # CAUTION: GPU resource management is critical here. Each task instantiates CrowTracker.
    # # If CrowTracker uses GPU models, num_workers should likely be 1 unless models can be
    # # distributed or CUDA_VISIBLE_DEVICES is managed per worker.
    # # For CPU-only models or if I/O is the bottleneck, cpu_count() // 2 might be reasonable.
    # num_workers = 1 # Default to 1 to be safe with GPU resources
    # logger.info(f"Potential for parallel processing with {num_workers} workers.")
    #
    # task_fn = partial(extract_crops_from_video_task,
    #                   base_dir_str=args.output_dir,
    #                   min_confidence_val=args.min_confidence,
    #                   frame_skip_val=args.frame_skip,
    #                   batch_size_frames_val=args.frame_batch_size,
    #                   enable_audio_val=args.enable_audio,
    #                   correct_orientation_val=args.correct_orientation,
    #                   target_frame_rate_val=args.target_fps)
    #
    # if num_workers > 1 and len(video_files_found) > 1:
    #     logger.info(f"Attempting to process videos in parallel with {num_workers} workers...")
    #     with Pool(processes=num_workers) as pool:
    #         all_summaries = list(tqdm(pool.imap_unordered(task_fn, video_files_found), total=len(video_files_found), desc="Parallel Video Processing"))
    #     logger.info("Parallel processing attempt finished.")
    # else:
    #     logger.info("Processing videos sequentially (num_workers=1 or single video).")
    #     all_summaries = []
    #     for video_path_str in video_files_found:
    #         # ... (sequential call as implemented above) ...
    #         pass # Already done sequentially above for this refactor

    total_frames_processed_all = sum(s.get('frames_processed_in_video', 0) for s in all_summaries)
    total_detections_all = sum(s.get('detections_count_in_video', 0) for s in all_summaries)
    succeeded_videos = sum(1 for s in all_summaries if s['status'] == 'success')
    
    logger.info(f"\n--- Overall Extraction Summary ---")
    logger.info(f"Successfully processed {succeeded_videos}/{len(video_files_found)} videos.")
    logger.info(f"Total frames processed across all videos: {total_frames_processed_all}")
    logger.info(f"Total detections found across all videos: {total_detections_all}")
    # Note: Crow counts are managed by CrowTracker globally, so a simple sum of "new crows per video" isn't accurate here.
    # Final crow count would be obtained by instantiating a final CrowTracker or checking its storage.
    logger.info("Extraction process finished.")
    logger.info(f"Check the '{args.output_dir}' directory for crops and metadata.")

if __name__ == "__main__":
    main()