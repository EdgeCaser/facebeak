import os
import argparse
import sys
from detection import detect_crows, detect_crows_cascade, detect_crows_parallel
from tracking import assign_crow_ids
from utils import extract_frames, save_video_with_labels
from db import get_all_crows, get_track_segments, update_track_segment_cluster
from crow_clustering import CrowClusterer
import cv2
from tqdm import tqdm
from datetime import datetime
import subprocess
import shutil
import traceback
import time

class TeeLogger:
    def __init__(self, log_path):
        # Open in write mode ('w') to start fresh each session
        self.log = open(log_path, 'w')
        self.stdout = sys.stdout
        self.stderr = sys.stderr
        sys.stdout = self
        sys.stderr = self
        
        # Add session start timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.write(f"Session Started at: {timestamp}\n")
        self.write(f"{'='*80}\n\n")
        
    def write(self, data):
        # Write and flush immediately to ensure continuous logging
        self.log.write(data)
        self.log.flush()  # Ensure data is written to disk immediately
        self.stdout.write(data)
        self.stdout.flush()
        
    def flush(self):
        self.log.flush()
        self.stdout.flush()
        
    def close(self):
        # Add session end timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.write(f"\n{'='*80}\n")
        self.write(f"Session Ended at: {timestamp}\n")
        
        sys.stdout = self.stdout
        sys.stderr = self.stderr
        self.log.close()

def get_video_orientation(cap):
    """Get video orientation from metadata."""
    try:
        # Try to get rotation from metadata
        rotation = int(cap.get(cv2.CAP_PROP_ORIENTATION_META))
        return rotation
    except:
        # If metadata not available, try to determine from video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # If height > width, video might be in portrait mode
        if height > width:
            return 90
        return 0

def rotate_frame(frame, rotation):
    """Rotate frame based on orientation."""
    if rotation == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    elif rotation == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    elif rotation == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame

def add_audio_to_video(processed_video, original_video, output_video):
    """Add audio from original video to processed video."""
    try:
        # Create temporary file in the same directory as output
        import tempfile
        import os
        output_dir = os.path.dirname(output_video)
        temp_output = os.path.join(output_dir, 'temp_with_audio.mp4')
        
        cmd = [
            'ffmpeg', '-y',
            '-i', processed_video,
            '-i', original_video,
            '-c', 'copy',
            '-map', '0:v:0',
            '-map', '1:a:0?',  # Make audio optional with ?
            '-shortest',
            temp_output
        ]
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        # If successful, move temp file to final output
        if os.path.exists(temp_output):
            os.replace(temp_output, output_video)
    except subprocess.CalledProcessError as e:
        print(f"[WARNING] Could not add audio: {e.stderr}")
        # If audio transfer fails, just copy the processed video to output
        import shutil
        shutil.copy2(processed_video, output_video)
    except Exception as e:
        print(f"[WARNING] Error during audio transfer: {str(e)}")
        # If any other error occurs, copy the processed video to output
        import shutil
        shutil.copy2(processed_video, output_video)

def compress_video(input_path, output_path, crf=23):
    """Compress video while maintaining quality."""
    try:
        # Create temporary file in the same directory as output
        import tempfile
        import os
        output_dir = os.path.dirname(output_path)
        temp_output = os.path.join(output_dir, 'temp_compressed.mp4')
        
        cmd = [
            'ffmpeg', '-y',
            '-i', input_path,
            '-vcodec', 'libx264',
            '-crf', str(crf),
            '-preset', 'medium',
            '-acodec', 'copy',
            temp_output
        ]
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        # If successful, move temp file to final output
        if os.path.exists(temp_output):
            os.replace(temp_output, output_path)
    except subprocess.CalledProcessError as e:
        print(f"[WARNING] Could not compress video: {e.stderr}")
        # If compression fails, just copy the input to output
        import shutil
        shutil.copy2(input_path, output_path)
    except Exception as e:
        print(f"[WARNING] Error during compression: {str(e)}")
        # If any other error occurs, copy the input to output
        import shutil
        shutil.copy2(input_path, output_path)

def check_ffmpeg():
    """Check if ffmpeg is installed and accessible."""
    ffmpeg_path = shutil.which('ffmpeg')
    if ffmpeg_path is None:
        print("\n[WARNING] ffmpeg is not installed or not found in PATH.")
        print("Audio transfer and video compression will not work.")
        print("\nTo install ffmpeg:")
        print("  Windows: Download from https://ffmpeg.org/download.html or use chocolatey: 'choco install ffmpeg'")
        print("  macOS: Use homebrew: 'brew install ffmpeg'")
        print("  Linux: Use package manager, e.g., 'sudo apt install ffmpeg' or 'sudo yum install ffmpeg'")
        print("\nAfter installing, restart the application.")
        return False
    return True

def process_video(video_path, output_path, detection_threshold=0.3, yolo_threshold=0.2, similarity_threshold=0.7, skip_frames=0, max_age=5, min_hits=2, iou_threshold=0.2, run_clustering=True):
    """
    Process a video file to detect and track crows, outputting both interpolated and frame-by-frame versions.
    """
    has_ffmpeg = check_ffmpeg()
    logger = TeeLogger('facebeak_session.log')
    try:
        print(f"[LOG] Starting facebeak session. Video: {video_path}, Output: {output_path}")
        print("[LOG] Parameters:")
        print(f"  - Frame skip: {skip_frames}")
        print(f"  - Detection threshold: {detection_threshold}")
        print(f"  - YOLO threshold: {yolo_threshold}")
        print(f"  - Max age: {max_age}")
        print(f"  - Min hits: {min_hits}")
        print(f"  - IOU threshold: {iou_threshold}")
        print(f"  - Embedding threshold: {similarity_threshold}")

        # Load all frames from video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[ERROR] Could not open video file: {video_path}")
            return

        rotation = get_video_orientation(cap)
        print(f"[INFO] Detected video rotation: {rotation} degrees")
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if rotation in [90, 270]:
            frame_width, frame_height = frame_height, frame_width
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"[INFO] Total frames in video: {total_frames}")

        # Read all frames
        all_frames = []
        with tqdm(total=total_frames, desc="Extracting frames") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = rotate_frame(frame, rotation)
                all_frames.append(frame)
                pbar.update(1)
        cap.release()
        print(f"[INFO] Read {len(all_frames)} frames")

        # Process frame-by-frame version (no skipping)
        print("[INFO] Processing frame-by-frame version...")
        frame_by_frame_detections = detect_crows_parallel(all_frames, score_threshold=detection_threshold, yolo_threshold=yolo_threshold)
        frame_by_frame_frames, _ = assign_crow_ids(
            all_frames,
            frame_by_frame_detections,
            video_path=video_path,
            max_age=max_age,
            min_hits=min_hits,
            iou_threshold=iou_threshold,
            embedding_threshold=similarity_threshold,
            return_track_history=True
        )

        # Process interpolated version (with frame skipping)
        if skip_frames > 0:
            print("[INFO] Processing interpolated version (with frame skipping)...")
            # Select frames for detection/tracking (every skip+1 frame)
            det_frames = [all_frames[i] for i in range(0, len(all_frames), skip_frames + 1)]
            det_indices = list(range(0, len(all_frames), skip_frames + 1))
            print(f"[INFO] Running detection/tracking on {len(det_frames)} frames")

            # Detect crows in det_frames
            detections_list = detect_crows_parallel(det_frames, score_threshold=detection_threshold, yolo_threshold=yolo_threshold)

            # Track crows across det_frames
            labeled_det_frames, track_history_per_frame = assign_crow_ids(
                det_frames,
                detections_list,
                video_path=video_path,
                max_age=max_age,
                min_hits=min_hits,
                iou_threshold=iou_threshold,
                embedding_threshold=similarity_threshold,
                return_track_history=True
            )

            # Interpolate crow boxes for skipped frames
            interp_track_boxes = [{} for _ in range(len(all_frames))]
            for i, det_idx in enumerate(det_indices):
                interp_track_boxes[det_idx] = track_history_per_frame[i]

            # For each track_id, interpolate its boxes across all frames
            track_ids = set()
            for frame_tracks in track_history_per_frame:
                track_ids.update(frame_tracks.keys())
            for track_id in track_ids:
                keypoints = [(det_indices[i], track_history_per_frame[i][track_id])
                            for i in range(len(det_indices)) if track_id in track_history_per_frame[i]]
                for idx in range(len(keypoints) - 1):
                    f0, box0 = keypoints[idx]
                    f1, box1 = keypoints[idx + 1]
                    for f in range(f0 + 1, f1):
                        alpha = (f - f0) / (f1 - f0)
                        interp_box = [
                            int((1 - alpha) * box0[j] + alpha * box1[j]) for j in range(4)
                        ]
                        interp_track_boxes[f][track_id] = interp_box

            # Draw crow boxes/IDs on all frames for interpolated version
            interpolated_frames = []
            for fidx, frame in enumerate(all_frames):
                frame_copy = frame.copy()
                for track_id, box in interp_track_boxes[fidx].items():
                    x1, y1, x2, y2 = box
                    label = f"Crow {track_id}"
                    cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    cv2.putText(frame_copy, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                interpolated_frames.append(frame_copy)
        else:
            # If no frame skipping, interpolated version is same as frame-by-frame
            interpolated_frames = frame_by_frame_frames

        # Save both versions
        print("[INFO] Saving output videos...")
        base_path = os.path.splitext(output_path)[0]
        save_video_with_labels(frame_by_frame_frames, f"{base_path}_frame_by_frame.mp4", video_path, output_mode='single')
        save_video_with_labels(interpolated_frames, f"{base_path}_interpolated.mp4", video_path, output_mode='single')

        print(f"[INFO] Processing complete. Outputs saved to:")
        print(f"  - Frame-by-frame version: {base_path}_frame_by_frame.mp4")
        print(f"  - Interpolated version: {base_path}_interpolated.mp4")

        # Print crow summary
        print("\n[INFO] Crow Summary:")
        crows = get_all_crows()
        for crow in crows:
            print(f"Crow {crow['id']} ({crow['name']}):")
            print(f"  - First seen: {crow['first_seen']}")
            print(f"  - Last seen: {crow['last_seen']}")
            print(f"  - Total sightings: {crow['total_sightings']}")
            print(f"  - Seen in {crow['video_count']} videos")
            print()

        # Run clustering if requested
        if run_clustering:
            print("\n[INFO] Running clustering analysis...")
            # Get segments for this video with minimum confidence
            segments = get_track_segments(min_confidence=0.5)
            video_segments = [s for s in segments if video_path in s['video_paths']]
            
            if video_segments:
                clusterer = CrowClusterer(eps=0.3, min_samples=3)
                clustered_segments = clusterer.cluster_segments(video_segments)
                
                if clustered_segments:
                    # Update cluster assignments in database
                    for segment in clustered_segments:
                        update_track_segment_cluster(segment['id'], segment['cluster_id'])
                    
                    # Group segments by cluster for display
                    clusters = {}
                    for segment in clustered_segments:
                        if segment['cluster_id'] not in clusters:
                            clusters[segment['cluster_id']] = []
                        clusters[segment['cluster_id']].append(segment['id'])
                    
                    print(f"\n[INFO] Found {len(clusters)} clusters:")
                    for cluster_id, segment_ids in clusters.items():
                        stats = clusterer.get_cluster_statistics(cluster_id)
                        print(f"\nCluster {cluster_id}:")
                        print(f"  - Segments: {len(segment_ids)}")
                        print(f"  - Average confidence: {stats.get('avg_confidence', 0):.2f}")
                        if 'toy_interactions' in stats:
                            print("  - Toy interactions:")
                            for toy, count in stats['toy_interactions'].items():
                                print(f"    * {toy}: {count} times")
                else:
                    print("[INFO] No clusters found in this video.")
            else:
                print("[INFO] No valid segments found for clustering in this video.")

    except Exception as e:
        print(f"[ERROR] An error occurred during processing: {str(e)}")
        print(f"[LOG] Session failed with error: {str(e)}")
        raise
    finally:
        logger.close()

def save_video_with_labels(frames, output_path, input_video_path=None, output_mode='single'):
    """
    Save video with labels and transfer audio if input video is provided.
    output_mode can be 'single' or 'both' (legacy support)
    """
    if not frames:
        print("[WARN] No frames to save!")
        return

    height, width = frames[0].shape[:2]
    fps = 30  # Default FPS
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    # Handle output paths
    if output_mode == 'both':
        base_path = os.path.splitext(output_path)[0]
        interpolated_path = f"{base_path}_interpolated.mp4"
        frame_by_frame_path = f"{base_path}_frame_by_frame.mp4"
        output_paths = [interpolated_path, frame_by_frame_path]
    else:
        output_paths = [output_path]

    # Process each output path
    for current_output in output_paths:
        temp_output = current_output + '.temp.mp4'
        
        # Delete existing files
        for path in [current_output, temp_output]:
            if os.path.exists(path):
                try:
                    os.remove(path)
                    print(f"[INFO] Deleted existing file: {path}")
                except Exception as e:
                    print(f"[ERROR] Could not delete existing file: {path}\n{e}")

        # Write frames to temp file
        out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))
        print(f"[INFO] Writing {len(frames)} frames to temp file: {temp_output}")
        for frame in frames:
            out.write(frame)
        out.release()
        print(f"[INFO] Finished writing temp file: {temp_output}")

        # Transfer audio if input video exists and ffmpeg is available
        if input_video_path and os.path.exists(input_video_path) and shutil.which('ffmpeg'):
            try:
                # First try with aac codec
                cmd = [
                    'ffmpeg', '-y',
                    '-i', temp_output,
                    '-i', input_video_path,
                    '-c:v', 'copy',
                    '-c:a', 'aac',
                    '-map', '0:v:0',
                    '-map', '1:a:0?',
                    '-shortest',
                    current_output
                ]
                print(f"[INFO] Attempting audio transfer with AAC codec: {' '.join(cmd)}")
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode != 0:
                    print(f"[WARNING] AAC codec failed, trying with copy codec: {result.stderr}")
                    # If aac fails, try with copy codec
                    cmd = [
                        'ffmpeg', '-y',
                        '-i', temp_output,
                        '-i', input_video_path,
                        '-c', 'copy',
                        '-map', '0:v:0',
                        '-map', '1:a:0?',
                        '-shortest',
                        current_output
                    ]
                    print(f"[INFO] Attempting audio transfer with copy codec: {' '.join(cmd)}")
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    
                    if result.returncode != 0:
                        raise subprocess.CalledProcessError(result.returncode, cmd, result.stdout, result.stderr)
                
                print(f"[INFO] Successfully transferred audio to: {current_output}")
                
                # Verify audio was transferred
                probe_cmd = [
                    'ffprobe', '-v', 'error',
                    '-select_streams', 'a',
                    '-show_entries', 'stream=codec_type',
                    '-of', 'default=noprint_wrappers=1:nokey=1',
                    current_output
                ]
                probe_result = subprocess.run(probe_cmd, capture_output=True, text=True)
                if 'audio' not in probe_result.stdout:
                    print("[WARNING] No audio stream found in output video")
                
            except subprocess.CalledProcessError as e:
                print(f"[WARNING] Error during audio transfer: {e.stderr}")
                print("[INFO] Copying temp file to output as fallback.")
                shutil.copy2(temp_output, current_output)
            except Exception as e:
                print(f"[WARNING] Error during audio transfer: {str(e)}\n{traceback.format_exc()}")
                print("[INFO] Copying temp file to output as fallback.")
                shutil.copy2(temp_output, current_output)
        else:
            if input_video_path and not shutil.which('ffmpeg'):
                print("[WARNING] ffmpeg not found - skipping audio transfer")
            print(f"[INFO] Renaming temp file to output file: {current_output}")
            os.rename(temp_output, current_output)

        # Clean up temp file
        for _ in range(3):
            if os.path.exists(temp_output):
                try:
                    os.remove(temp_output)
                    print(f"[INFO] Deleted temp file: {temp_output}")
                    break
                except Exception as e:
                    print(f"[WARNING] Could not delete temp file (attempt): {e}")
                    time.sleep(0.5)
            else:
                break

        if os.path.exists(temp_output):
            print(f"[WARNING] Temp file still exists after cleanup: {temp_output}")
        if os.path.exists(current_output):
            print(f"[INFO] Output video ready: {current_output}")
        else:
            print(f"[ERROR] Output video was not created: {current_output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect and track crows in video")
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument("--output", required=True, help="Path to output video file")
    parser.add_argument("--detection-threshold", type=float, default=0.3, help="Detection confidence threshold")
    parser.add_argument("--yolo-threshold", type=float, default=0.2, help="YOLO detection threshold")
    parser.add_argument("--max-age", type=int, default=5, help="Maximum number of frames to keep alive a track without associated detections")
    parser.add_argument("--min-hits", type=int, default=2, help="Minimum number of associated detections before track is initialised")
    parser.add_argument("--iou-threshold", type=float, default=0.2, help="Minimum IOU for match")
    parser.add_argument("--embedding-threshold", type=float, default=0.7, help="Similarity threshold for crow matching")
    parser.add_argument("--skip", type=int, default=0, help="Number of frames to skip between detections")
    parser.add_argument("--no-clustering", action="store_true", help="Skip clustering analysis")
    parser.add_argument("--output-mode", choices=['both', 'interpolated', 'frame_by_frame'], 
                       default='both', help="Output mode for processed videos")
    args = parser.parse_args()

    process_video(
        args.video,
        args.output,
        args.detection_threshold,
        args.yolo_threshold,
        args.embedding_threshold,
        args.skip,
        max_age=args.max_age,
        min_hits=args.min_hits,
        iou_threshold=args.iou_threshold,
        run_clustering=not args.no_clustering
    )
