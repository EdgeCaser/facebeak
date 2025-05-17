import os
import argparse
import sys
from detection import detect_crows, detect_crows_cascade, detect_crows_parallel
from tracking import assign_crow_ids
from utils import extract_frames, save_video_with_labels
from db import get_all_crows
import cv2
from tqdm import tqdm
from datetime import datetime
import subprocess

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

def process_video(video_path, output_path, detection_threshold=0.3, yolo_threshold=0.2, similarity_threshold=0.7, skip_frames=0, max_age=5, min_hits=2, iou_threshold=0.2):
    """
    Process a video file to detect and track crows.
    """
    # Initialize logger
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

        print(f"[INFO] Processing video: {video_path}")
        print(f"[INFO] Output will be saved to: {output_path}")
        print(f"[INFO] Using detection threshold: {detection_threshold}")
        print(f"[INFO] Using YOLO threshold: {yolo_threshold}")
        print(f"[INFO] Using similarity threshold: {similarity_threshold}")
        print(f"[INFO] Frame skip: {skip_frames} (processing every {skip_frames + 1} frame)")

        # Load video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[ERROR] Could not open video file: {video_path}")
            return

        # Get video orientation
        rotation = get_video_orientation(cap)
        print(f"[INFO] Detected video rotation: {rotation} degrees")

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Adjust dimensions if video is rotated
        if rotation in [90, 270]:
            frame_width, frame_height = frame_height, frame_width
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"[INFO] Total frames in video: {total_frames}")
        
        # Calculate expected number of frames to process
        frames_to_process = total_frames // (skip_frames + 1)
        print(f"[INFO] Will process approximately {frames_to_process} frames")

        # Initialize video writer with adjusted dimensions
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

        frames = []
        frame_count = 0
        processed_frames = 0

        print("[INFO] Reading video frames...")
        with tqdm(total=total_frames, desc="Extracting frames") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % (skip_frames + 1) == 0:
                    # Rotate frame if needed
                    frame = rotate_frame(frame, rotation)
                    frames.append(frame)
                    processed_frames += 1
                frame_count += 1
                pbar.update(1)

        cap.release()
        print(f"[INFO] Read {len(frames)} frames for processing")
        print(f"[LOG] Extracted {len(frames)} frames.")

        # Detect crows in frames
        print("[INFO] Detecting crows...")
        detections_list = detect_crows_parallel(frames, score_threshold=detection_threshold, yolo_threshold=yolo_threshold)

        # Track crows across frames
        print("[INFO] Tracking crows...")
        labeled_frames = assign_crow_ids(
            frames, 
            detections_list,
            video_path=video_path,  # Pass video path for database tracking
            max_age=max_age,
            min_hits=min_hits,
            iou_threshold=iou_threshold,
            embedding_threshold=similarity_threshold
        )

        # Write output video
        print("[INFO] Writing output video...")
        temp_output = os.path.join(os.path.dirname(output_path), 'temp_processed.mp4')
        out = cv2.VideoWriter(temp_output, fourcc, fps, (frame_width, frame_height))
        
        for frame in tqdm(labeled_frames, desc="Saving video"):
            out.write(frame)
        out.release()
        
        # Add audio and compress
        print("[INFO] Adding audio to output video...")
        add_audio_to_video(temp_output, video_path, output_path)
        
        # Clean up temporary file
        if os.path.exists(temp_output):
            os.remove(temp_output)
            
        print(f"[INFO] Processing complete. Output saved to: {output_path}")

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

    except Exception as e:
        print(f"[ERROR] An error occurred during processing: {str(e)}")
        print(f"[LOG] Session failed with error: {str(e)}")
        # Clean up any temporary files
        if 'temp_output' in locals() and os.path.exists(temp_output):
            try:
                os.remove(temp_output)
            except:
                pass
        raise
    finally:
        logger.close()

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
    args = parser.parse_args()

    process_video(
        args.video,
        args.output,
        args.detection_threshold,
        args.yolo_threshold,
        args.embedding_threshold,  # Using embedding_threshold as similarity_threshold
        args.skip,
        max_age=args.max_age,
        min_hits=args.min_hits,
        iou_threshold=args.iou_threshold
    )
