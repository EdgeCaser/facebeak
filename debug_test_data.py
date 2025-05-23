import sys
from pathlib import Path
import tempfile
import cv2
import json
import subprocess

# Add the current directory to the path
sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, str(Path.cwd() / "tests"))

# Import required modules
from detection import detect_crows_parallel
from tracking import EnhancedTracker

def debug_video_test_data():
    """Debug version of video_test_data fixture."""
    # Create base directory
    base_dir = Path(tempfile.mkdtemp(prefix="test_data"))
    print(f"Created temp directory: {base_dir}")

    # Get the test videos directory
    test_videos_dir = Path("../unit_testing_videos")
    print(f"[DEBUG] test_videos_dir is {test_videos_dir}")
    print(f"[DEBUG] test_videos_dir exists: {test_videos_dir.exists()}")
    
    if not test_videos_dir.exists():
        print("[DEBUG] test_videos_dir does not exist, skipping.")
        return {"base_dir": base_dir, "valid": False}

    # List videos
    videos = list(test_videos_dir.glob("*.mp4"))
    print(f"[DEBUG] Found videos: {videos}")

    # Initialize detection models
    tracker = EnhancedTracker()

    # Initialize metadata dictionary
    metadata = {}
    valid_videos_found = False

    # Process first video only for debugging
    if videos:
        video_path = videos[0]
        crow_id = video_path.stem.split("_")[0]
        
        print(f"[DEBUG] processing video {video_path} with crow_id {crow_id}")
        crow_dir = base_dir / crow_id
        images_dir = crow_dir / "images"
        audio_dir = crow_dir / "audio"
        images_dir.mkdir(parents=True, exist_ok=True)
        audio_dir.mkdir(exist_ok=True)

        # Initialize metadata for this crow
        metadata[crow_id] = {"images": {}}

        # Open video file
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"[DEBUG] could not open video {video_path}")
        else:
            valid_videos_found = True
            print(f"[DEBUG] Successfully opened video {video_path}")
            
            # Try to read a few frames
            frame_count = 0
            while frame_count < 5:
                ret, frame = cap.read()
                if not ret:
                    print(f"[DEBUG] Could not read frame {frame_count}")
                    break
                print(f"[DEBUG] Read frame {frame_count}, shape: {frame.shape}")
                frame_count += 1
            
            cap.release()

    return {
        "base_dir": base_dir,
        "valid": valid_videos_found,
        "metadata": metadata
    }

# Run the debug function
data = debug_video_test_data()

print(f"Video test data valid: {data['valid']}")
print(f"Base directory: {data['base_dir']}")

if data['valid']:
    print(f"Metadata: {data['metadata']}")
    print("\nDirectory structure:")
    for item in data['base_dir'].rglob("*"):
        if item.is_file():
            print(f"  FILE: {item.relative_to(data['base_dir'])}")
        else:
            print(f"  DIR:  {item.relative_to(data['base_dir'])}/")
else:
    print("No valid videos found") 