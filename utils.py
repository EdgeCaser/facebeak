import cv2
from tqdm import tqdm
import os

def extract_frames(video_path, skip=5, progress_callback=None):
    """Extract frames from a video file.
    
    Args:
        video_path: Path to the video file
        skip: Number of frames to skip between extractions
        progress_callback: Optional callback function to report progress (0.0 to 1.0)
    
    Returns:
        List of extracted frames
        
    Raises:
        FileNotFoundError: If video file does not exist
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
        
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
        
    # Get total frame count for progress bar
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    
    print(f"[INFO] Total frames in video: {total_frames}")
    print(f"[INFO] Processing every {skip}th frame...")
    
    with tqdm(total=total_frames, desc="Extracting frames") as pbar:
        i = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if i % skip == 0:
                frames.append(frame)
            i += 1
            pbar.update(1)
            if progress_callback:
                progress_callback(i / total_frames)
    
    cap.release()
    print(f"[INFO] Extracted {len(frames)} frames")
    return frames

def save_video_with_labels(frames, labels, output_path, fps=15, progress_callback=None):
    """Save frames as a video with optional labels.
    
    Args:
        frames: List of frames to save
        labels: List of label dictionaries for each frame
        output_path: Path to save the video
        fps: Frames per second
        progress_callback: Optional callback function to report progress (0.0 to 1.0)
        
    Raises:
        ValueError: If frames list is empty or frames/labels lengths don't match
        KeyError: If label dictionaries are missing required fields
    """
    if not frames:
        raise ValueError("No frames to save")
        
    if labels and len(frames) != len(labels):
        raise ValueError("Number of frames and labels must match")
        
    # Validate label format
    if labels:
        required_fields = {'bbox', 'label'}
        for frame_labels in labels:
            for label in frame_labels:
                if not all(field in label for field in required_fields):
                    raise KeyError(f"Label missing required fields: {required_fields}")

    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"[INFO] Saving {len(frames)} frames to {output_path}")
    for i, frame in enumerate(tqdm(frames, desc="Saving video")):
        # Draw labels if provided
        if labels:
            for label in labels[i]:
                bbox = label['bbox']
                text = label['label']
                cv2.rectangle(frame, 
                            (int(bbox[0]), int(bbox[1])),
                            (int(bbox[2]), int(bbox[3])),
                            (0, 255, 0), 2)
                cv2.putText(frame, text,
                           (int(bbox[0]), int(bbox[1] - 5)),
                           cv2.FONT_HERSHEY_SIMPLEX,
                           0.5, (0, 255, 0), 2)
        
        out.write(frame)
        if progress_callback:
            progress_callback((i + 1) / len(frames))
            
    out.release()
    print("[INFO] Video save complete")
