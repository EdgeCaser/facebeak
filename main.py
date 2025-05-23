import os
import argparse
import sys
from detection import detect_crows_legacy, detect_crows_cascade, detect_crows_parallel
from tracking import assign_crow_ids, Sort
from utils import extract_frames, save_video_with_labels
from db import get_all_crows
import cv2
from tqdm import tqdm
from datetime import datetime
import subprocess
import numpy as np
from typing import List

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
        # Use Popen to capture and log output in real-time
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in process.stdout:
            print(line.strip())  # This will be captured by TeeLogger
        process.wait()
        
        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, cmd)
        
        # If successful, move temp file to final output
        if os.path.exists(temp_output):
            os.replace(temp_output, output_video)
    except subprocess.CalledProcessError as e:
        print(f"[WARNING] Could not add audio: {str(e)}")
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
        # Use Popen to capture and log output in real-time
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in process.stdout:
            print(line.strip())  # This will be captured by TeeLogger
        process.wait()
        
        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, cmd)
        
        # If successful, move temp file to final output
        if os.path.exists(temp_output):
            os.replace(temp_output, output_path)
    except subprocess.CalledProcessError as e:
        print(f"[WARNING] Could not compress video: {str(e)}")
        # If compression fails, just copy the input to output
        import shutil
        shutil.copy2(input_path, output_path)
    except Exception as e:
        print(f"[WARNING] Error during compression: {str(e)}")
        # If any other error occurs, copy the input to output
        import shutil
        shutil.copy2(input_path, output_path)

def interpolate_frames(processed_frames, total_frames, track_history_per_frame=None):
    """Interpolate between processed frames to reconstruct a full-frame video."""
    if len(processed_frames) == 0:
        return []
    
    # Calculate frames per segment (including skipped frames)
    frames_per_segment = total_frames // (len(processed_frames) - 1) if len(processed_frames) > 1 else total_frames
    
    interp_frames = []
    for i in range(len(processed_frames) - 1):
        # Add the current processed frame
        interp_frames.append(processed_frames[i])
        
        # Get current and next frame's track information
        current_tracks = track_history_per_frame[i] if track_history_per_frame else {}
        next_tracks = track_history_per_frame[i + 1] if track_history_per_frame else {}
        
        # Interpolate frames between current and next
        for g in range(1, frames_per_segment):
            alpha = g / frames_per_segment
            interp_frame = processed_frames[i].copy()
            
            # Interpolate each track's bounding box
            for track_id in set(current_tracks.keys()) | set(next_tracks.keys()):
                if track_id in current_tracks and track_id in next_tracks:
                    # Both frames have this track, interpolate position
                    curr_box = current_tracks[track_id]
                    next_box = next_tracks[track_id]
                    
                    # Linear interpolation of box coordinates
                    interp_box = [
                        int(curr_box[0] * (1 - alpha) + next_box[0] * alpha),
                        int(curr_box[1] * (1 - alpha) + next_box[1] * alpha),
                        int(curr_box[2] * (1 - alpha) + next_box[2] * alpha),
                        int(curr_box[3] * (1 - alpha) + next_box[3] * alpha)
                    ]
                    
                    # Draw interpolated box
                    cv2.rectangle(interp_frame, (interp_box[0], interp_box[1]), 
                                (interp_box[2], interp_box[3]), (0, 255, 255), 2)
                    label = f"Crow {track_id}"
                    cv2.putText(interp_frame, label, (interp_box[0], interp_box[1]-10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                elif track_id in current_tracks:
                    # Track only in current frame, fade out
                    if alpha < 0.5:  # Only show in first half of interpolation
                        box = current_tracks[track_id]
                        opacity = 1 - (alpha * 2)  # Fade from 1 to 0
                        cv2.rectangle(interp_frame, (box[0], box[1]), (box[2], box[3]), 
                                    (0, int(255 * opacity), int(255 * opacity)), 2)
                        cv2.putText(interp_frame, f"Crow {track_id}", (box[0], box[1]-10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.9, 
                                  (0, int(255 * opacity), int(255 * opacity)), 2)
                elif track_id in next_tracks:
                    # Track only in next frame, fade in
                    if alpha >= 0.5:  # Only show in second half of interpolation
                        box = next_tracks[track_id]
                        opacity = (alpha - 0.5) * 2  # Fade from 0 to 1
                        cv2.rectangle(interp_frame, (box[0], box[1]), (box[2], box[3]), 
                                    (0, int(255 * opacity), int(255 * opacity)), 2)
                        cv2.putText(interp_frame, f"Crow {track_id}", (box[0], box[1]-10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.9, 
                                  (0, int(255 * opacity), int(255 * opacity)), 2)
            
            interp_frames.append(interp_frame)
    
    # Add the last processed frame
    interp_frames.append(processed_frames[-1])
    
    # Ensure we have exactly total_frames
    if len(interp_frames) > total_frames:
        interp_frames = interp_frames[:total_frames]
    elif len(interp_frames) < total_frames:
        # Pad with the last frame if needed
        interp_frames.extend([processed_frames[-1]] * (total_frames - len(interp_frames)))
    
    return interp_frames

def parse_args(args=None):
    parser = argparse.ArgumentParser(description="Process video for crow detection and tracking")
    parser.add_argument("--video", required=True, help="Input video file")
    parser.add_argument("--skip-output", required=True, help="Output video file for frame-skipped detection")
    parser.add_argument("--full-output", required=True, help="Output video file for full-frame interpolated tracking")
    parser.add_argument("--detection-threshold", type=float, default=0.3, help="Detection confidence threshold")
    parser.add_argument("--yolo-threshold", type=float, default=0.2, help="YOLO confidence threshold")
    parser.add_argument("--max-age", type=int, default=5, help="Maximum age of a track")
    parser.add_argument("--min-hits", type=int, default=2, help="Minimum hits to start tracking")
    parser.add_argument("--iou-threshold", type=float, default=0.2, help="IOU threshold for tracking")
    parser.add_argument("--embedding-threshold", type=float, default=0.7, help="Embedding similarity threshold")
    parser.add_argument("--skip", type=int, default=5, help="Number of frames to skip")
    parser.add_argument("--multi-view-stride", type=int, default=1, help="Stride for multi-view extraction")
    parser.add_argument("--preserve-audio", action="store_true", help="Preserve audio in output videos")
    return parser.parse_args(args)

def calculate_iou(box1, box2):
    """Calculate Intersection over Union between two bounding boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = box1_area + box2_area - intersection
    
    return intersection / union if union > 0 else 0

def draw_detections_and_tracks(frame, detections, tracks):
    """Draw detections and tracks on the frame."""
    # Draw detections if provided
    if detections:
        for det in detections:
            box = det['bbox']
            score = det['score']
            label = f"{det['class']} {score:.2f}"
            
            # Draw box
            cv2.rectangle(frame, 
                        (int(box[0]), int(box[1])), 
                        (int(box[2]), int(box[3])), 
                        (0, 255, 0), 2)
            
            # Draw label
            cv2.putText(frame, label, 
                       (int(box[0]), int(box[1]-10)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, 
                       (0, 255, 0), 2)
    
    # Draw tracks if provided
    if tracks is not None and len(tracks) > 0:
        for track in tracks:
            # Get box coordinates and track ID
            x1, y1, x2, y2, track_id = track[:5]
            
            # Draw box
            cv2.rectangle(frame, 
                        (int(x1), int(y1)), 
                        (int(x2), int(y2)), 
                        (0, 255, 255), 2)
            
            # Draw track ID
            label = f"Crow {int(track_id)}"
            cv2.putText(frame, label, 
                       (int(x1), int(y1-10)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, 
                       (0, 255, 255), 2)
    
    return frame

def process_video(video_path: str, skip_output: str, full_output: str, args):
    """Process video for crow detection and tracking."""
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate effective FPS for skip-frame video
    skip_fps = fps / args.skip if args.skip > 1 else fps
    
    # Initialize video writers
    skip_writer = cv2.VideoWriter(
        skip_output,
        cv2.VideoWriter_fourcc(*'mp4v'),
        skip_fps,
        (width, height)
    )
    
    full_writer = cv2.VideoWriter(
        full_output,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,  # Use original FPS for full-frame video
        (width, height)
    )
    
    # Initialize tracker
    tracker = Sort(max_age=args.max_age, min_hits=args.min_hits, iou_threshold=args.iou_threshold)
    
    # Process frames
    frame_count = 0
    processed_frames = []
    processed_tracks = []
    frames_to_process = []
    frame_indices = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % args.skip == 0:
            frames_to_process.append(frame)
            frame_indices.append(frame_count)
            
        frame_count += 1
    
    # Process frames in batches
    if frames_to_process:
        detections_list = detect_crows_parallel(
            frames_to_process,
            score_threshold=args.detection_threshold,
            yolo_threshold=args.yolo_threshold
        )
        
        # Process each frame's detections
        for frame, detections, frame_idx in zip(frames_to_process, detections_list, frame_indices):
            # Convert detections to format expected by tracker
            dets = np.array([[d['bbox'][0], d['bbox'][1], d['bbox'][2], d['bbox'][3], d['score']] 
                            for d in detections]) if detections else np.empty((0, 5))
            
            # Update tracker
            tracks = tracker.update(dets)
            
            # Draw detections and tracks
            processed_frame = draw_detections_and_tracks(frame.copy(), detections, tracks)
            skip_writer.write(processed_frame)
            
            # Store for interpolation
            processed_frames.append(frame_idx)
            processed_tracks.append(tracks)
    
    # Release video capture and skip-frame writer
    cap.release()
    skip_writer.release()
    
    # Generate full-frame video with interpolated tracks
    interpolate_frames(
        video_path,
        full_output,
        processed_frames,
        processed_tracks,
        fps,
        args.preserve_audio
    )
    
    return frame_count

def interpolate_frames(video_path: str, output_path: str, processed_frames: List[int], 
                      processed_tracks: List[np.ndarray], fps: float, preserve_audio: bool):
    """Generate full-frame video with interpolated tracks."""
    if not processed_frames or not processed_tracks:
        print("[WARNING] No processed frames or tracks available for interpolation")
        # Just copy the original video if no tracks to interpolate
        import shutil
        shutil.copy2(video_path, output_path)
        return

    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create temporary video without audio
    temp_output = output_path + ".temp.mp4"
    writer = cv2.VideoWriter(
        temp_output,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (width, height)
    )
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        try:
            # Find nearest processed frames
            prev_frames = [i for i, f in enumerate(processed_frames) if f <= frame_count]
            next_frames = [i for i, f in enumerate(processed_frames) if f >= frame_count]
            
            if not prev_frames and not next_frames:
                # No processed frames available, use original frame
                writer.write(frame)
                frame_count += 1
                continue
                
            if not prev_frames:
                # Only future frames available, use the first one
                next_idx = next_frames[0]
                tracks = processed_tracks[next_idx]
            elif not next_frames:
                # Only past frames available, use the last one
                prev_idx = prev_frames[-1]
                tracks = processed_tracks[prev_idx]
            else:
                # Both past and future frames available
                prev_idx = prev_frames[-1]
                next_idx = next_frames[0]
                
                if prev_idx == next_idx:
                    # Use exact track
                    tracks = processed_tracks[prev_idx]
                else:
                    # Interpolate between tracks
                    prev_tracks = processed_tracks[prev_idx]
                    next_tracks = processed_tracks[next_idx]
                    prev_frame = processed_frames[prev_idx]
                    next_frame = processed_frames[next_idx]
                    
                    # Interpolate tracks
                    alpha = (frame_count - prev_frame) / (next_frame - prev_frame)
                    tracks = interpolate_tracks(prev_tracks, next_tracks, alpha)
            
            # Draw interpolated tracks
            frame = draw_detections_and_tracks(frame, None, tracks)
            writer.write(frame)
            
        except Exception as e:
            print(f"[WARNING] Error processing frame {frame_count}: {str(e)}")
            # Write original frame if there's an error
            writer.write(frame)
            
        frame_count += 1
    
    # Release resources
    cap.release()
    writer.release()
    
    if preserve_audio:
        # Use ffmpeg to combine video with original audio
        try:
            cmd = [
                'ffmpeg', '-y',
                '-i', temp_output,
                '-i', video_path,
                '-c:v', 'copy',
                '-c:a', 'aac',
                '-map', '0:v:0',
                '-map', '1:a:0',
                output_path
            ]
            # Use Popen to capture and log output in real-time
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in process.stdout:
                print(line.strip())  # This will be captured by TeeLogger
            process.wait()
            
            if process.returncode != 0:
                raise subprocess.CalledProcessError(process.returncode, cmd)
                
            os.remove(temp_output)
        except Exception as e:
            print(f"[WARNING] Could not add audio: {str(e)}")
            os.rename(temp_output, output_path)
    else:
        os.rename(temp_output, output_path)

def interpolate_tracks(prev_tracks: np.ndarray, next_tracks: np.ndarray, alpha: float) -> np.ndarray:
    """Interpolate between two sets of tracks."""
    if len(prev_tracks) == 0:
        return next_tracks
    if len(next_tracks) == 0:
        return prev_tracks
    
    # Match tracks between frames
    matched_tracks = []
    for prev_track in prev_tracks:
        best_iou = 0
        best_next = None
        for next_track in next_tracks:
            iou = calculate_iou(prev_track[:4], next_track[:4])
            if iou > best_iou:
                best_iou = iou
                best_next = next_track
        
        if best_next is not None and best_iou > 0.3:  # Minimum IOU threshold
            # Interpolate bounding box
            interp_box = prev_track[:4] * (1 - alpha) + best_next[:4] * alpha
            # Keep ID from previous track
            interp_track = np.concatenate([interp_box, prev_track[4:]])
            matched_tracks.append(interp_track)
    
    return np.array(matched_tracks) if matched_tracks else np.array([])

def process_frame(frame, detections, tracker):
    """
    Process a single frame: update tracker, draw detections and tracks.
    Returns (processed_frame, tracks)
    """
    # Convert detections to format expected by tracker
    dets = np.array([[d['bbox'][0], d['bbox'][1], d['bbox'][2], d['bbox'][3], d['score']] 
                    for d in detections]) if detections else np.empty((0, 5))
    # Update tracker
    tracks = tracker.update(dets)
    # Draw detections and tracks
    processed_frame = draw_detections_and_tracks(frame.copy(), detections, tracks)
    return processed_frame, tracks

if __name__ == "__main__":
    # Initialize logger
    log_path = "facebeak_Session.log"
    logger = TeeLogger(log_path)
    
    try:
        args = parse_args()
        process_video(
            args.video,
            args.skip_output,
            args.full_output,
            args
        )
    except Exception as e:
        print(f"[ERROR] An error occurred: {str(e)}")
        raise
    finally:
        # Ensure logger is properly closed
        logger.close()
