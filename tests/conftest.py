import os
import pytest
import torch
import numpy as np
from pathlib import Path
import shutil
import tempfile
import cv2
from PIL import Image
import soundfile as sf
import librosa
import subprocess
import json

def create_dummy_image(size=(224, 224), num_channels=3):
    """Create a dummy image for testing."""
    # Create a more realistic dummy image with a crow-like shape
    img = np.zeros((size[0], size[1], num_channels), dtype=np.uint8)
    
    # Add a dark silhouette (crow-like shape)
    center_x, center_y = size[1] // 2, size[0] // 2
    radius = min(size) // 3
    
    # Create a dark oval for the body
    cv2.ellipse(img, (center_x, center_y), (radius, radius//2), 0, 0, 360, (50, 50, 50), -1)
    
    # Add a smaller circle for the head
    head_radius = radius // 2
    cv2.circle(img, (center_x + radius//2, center_y - radius//4), head_radius, (50, 50, 50), -1)
    
    # Add some noise to make it more realistic
    noise = np.random.normal(0, 10, img.shape).astype(np.uint8)
    img = cv2.add(img, noise)
    
    return Image.fromarray(img)

def create_dummy_audio(duration=1.0, sr=16000):
    """Create a dummy audio signal for testing."""
    # Create a more realistic crow-like audio signal
    t = np.linspace(0, duration, int(sr * duration))
    
    # Base frequency for crow-like sound (around 1-2 kHz)
    base_freq = 1500
    
    # Create a modulated signal to simulate crow cawing
    signal = np.sin(2 * np.pi * base_freq * t)
    
    # Add amplitude modulation
    am = 0.5 * (1 + np.sin(2 * np.pi * 5 * t))  # 5 Hz modulation
    signal = signal * am
    
    # Add some harmonics
    signal += 0.3 * np.sin(2 * np.pi * base_freq * 2 * t)  # First harmonic
    signal += 0.2 * np.sin(2 * np.pi * base_freq * 3 * t)  # Second harmonic
    
    # Add some noise
    noise = np.random.normal(0, 0.1, signal.shape)
    signal = signal + noise
    
    # Normalize
    signal = signal / np.max(np.abs(signal))
    
    return signal.astype(np.float32)

@pytest.fixture(scope="session")
def test_data_dir(tmp_path_factory):
    """Create a temporary directory with test data structure."""
    test_dir = tmp_path_factory.mktemp("test_data")
    
    # Create directory structure for multiple crows
    for crow_id in ["crow1", "crow2", "crow3"]:
        # Create image directory
        img_dir = test_dir / crow_id / "images"
        img_dir.mkdir(parents=True)
        
        # Create video directory
        video_dir = test_dir / crow_id / "videos"
        video_dir.mkdir(parents=True)
        
        # Create 5 images for each crow
        for i in range(5):
            img_path = img_dir / f"{crow_id}_img_{i}.jpg"
            img = create_dummy_image()
            img.save(img_path)
            
            # Create corresponding video file
            video_path = video_dir / f"{crow_id}_img_{i}.mp4"
            create_dummy_video(video_path, num_frames=30, fps=30)
    
    # Create a metadata file with timestamps
    metadata = {
        "crow1": {
            "images": {
                f"crow1_img_{i}.jpg": {"timestamp": f"2024-03-19 10:00:{i:02d}"} 
                for i in range(5)
            }
        },
        "crow2": {
            "images": {
                f"crow2_img_{i}.jpg": {"timestamp": f"2024-03-19 10:01:{i:02d}"}
                for i in range(5)
            }
        },
        "crow3": {
            "images": {
                f"crow3_img_{i}.jpg": {"timestamp": f"2024-03-19 10:02:{i:02d}"}
                for i in range(5)
            }
        }
    }
    
    with open(test_dir / "metadata.json", "w") as f:
        json.dump(metadata, f)
    
    return test_dir

def create_dummy_video(path, num_frames=30, fps=30):
    """Create a dummy video file for testing."""
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(path), fourcc, fps, (640, 480))
    
    # Write frames with a crow-like shape
    for i in range(num_frames):
        # Create frame with a crow-like shape
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Add a dark silhouette (crow-like shape)
        center_x, center_y = 320, 240
        radius = 100
        
        # Create a dark oval for the body
        cv2.ellipse(frame, (center_x, center_y), (radius, radius//2), 0, 0, 360, (50, 50, 50), -1)
        
        # Add a smaller circle for the head
        head_radius = radius // 2
        cv2.circle(frame, (center_x + radius//2, center_y - radius//4), head_radius, (50, 50, 50), -1)
        
        # Add some noise to make it more realistic
        noise = np.random.normal(0, 10, frame.shape).astype(np.uint8)
        frame = cv2.add(frame, noise)
        
        out.write(frame)
    
    out.release()

@pytest.fixture(scope="session")
def device():
    """Get the device to use for testing."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

@pytest.fixture(scope="session")
def sample_batch():
    """Create a sample batch of data for testing."""
    batch_size = 4
    img_size = (3, 224, 224)  # RGB image
    audio_size = (2, 128, 128)  # 2 channels (mel_spec and chroma)
    
    # Create dummy image data
    anchor_imgs = torch.randn(batch_size, *img_size)
    pos_imgs = torch.randn(batch_size, *img_size)
    neg_imgs = torch.randn(batch_size, *img_size)
    
    # Create dummy audio data
    anchor_audio = torch.randn(batch_size, *audio_size)
    pos_audio = torch.randn(batch_size, *audio_size)
    neg_audio = torch.randn(batch_size, *audio_size)
    
    return {
        'imgs': (anchor_imgs, pos_imgs, neg_imgs),
        'audio': (anchor_audio, pos_audio, neg_audio),
        'labels': torch.randint(0, 2, (batch_size,))  # Dummy labels
    }

@pytest.fixture(scope="session")
def model_config():
    """Get model configuration for testing."""
    return {
        "visual_dim": 512,
        "audio_dim": 512,
        "hidden_dim": 1024,
        "output_dim": 512
    }

# Add a marker for video tests
def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "video: mark test as requiring video data (will be skipped if no videos available)"
    )

@pytest.fixture(scope="session")
def video_test_data(tmp_path_factory):
    """Create a test dataset from videos by extracting frames and audio."""
    # Create base directory
    base_dir = tmp_path_factory.mktemp("test_data")

    # Get the test videos directory
    test_videos_dir = Path("unit_testing_videos")
    print("[DEBUG] video_test_data: test_videos_dir is", test_videos_dir)
    if not test_videos_dir.exists():
        print("[DEBUG] video_test_data: test_videos_dir does not exist, skipping.")
        return {"base_dir": base_dir, "valid": False}

    # Initialize detection models
    from detection import detect_crows_parallel
    from tracking import extract_crow_image

    # Initialize metadata dictionary
    metadata = {}
    valid_videos_found = False

    # Process each video in the test directory
    for video_path in test_videos_dir.glob("*.mp4"):
        # Extract crow_id from filename (assuming format: "crow_id_*.mp4")
        crow_id = video_path.stem.split("_")[0]
        
        print("[DEBUG] video_test_data: processing video", video_path, "with crow_id", crow_id)
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
            print("[DEBUG] video_test_data: could not open video", video_path)
            continue

        valid_videos_found = True
        # Extract frames in batches for detection
        frames = []
        frame_numbers = []
        frame_count = 0
        max_frames = 30  # Limit to 30 frames per video for testing
        img_count = 0  # Counter for saved images
        
        while frame_count < max_frames and img_count < 5:  # Save up to 5 images per crow
            frames = []
            frame_numbers = []
            
            # Read a batch of frames
            for _ in range(10):  # Process 10 frames at a time
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
                frame_numbers.append(frame_count)
                frame_count += 1
                
            if not frames:
                break
                
            # Detect crows in the batch
            try:
                print(f"[DEBUG] video_test_data: detecting crows in batch of {len(frames)} frames")
                detections = detect_crows_parallel(frames, score_threshold=0.3)
                
                # Process detections and save frames
                for frame_idx, frame_dets in enumerate(detections):
                    if not frame_dets:
                        continue
                        
                    frame = frames[frame_idx]
                    frame_num = frame_numbers[frame_idx]
                    
                    for det in frame_dets:
                        if det['score'] < 0.3:  # Skip low confidence detections
                            continue
                            
                        # Extract crop using extract_crow_image
                        crop = extract_crow_image(frame, det['bbox'])
                        if crop is None:
                            print(f"[DEBUG] video_test_data: failed to extract crop for detection {det}")
                            continue
                            
                        # Save the crop
                        crop_np = (crop['full'].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                        img_path = images_dir / f"frame_{frame_num:06d}.jpg"
                        cv2.imwrite(str(img_path), crop_np)
                        
                        # Update metadata
                        metadata[crow_id]["images"][str(img_path)] = {
                            "frame": frame_num,
                            "bbox": det['bbox'].tolist(),
                            "score": float(det['score'])
                        }
                        img_count += 1
                        
                        if img_count >= 5:  # Stop after saving 5 images
                            break
                            
                    if img_count >= 5:
                        break
                        
            except Exception as e:
                print(f"[ERROR] video_test_data: error processing batch: {str(e)}")
                continue

        cap.release()

        # If no images were saved for this crow, save up to 5 raw frames
        if img_count == 0:
            print(f"[DEBUG] video_test_data: no detections for {crow_id}, saving raw frames instead")
            cap_raw = cv2.VideoCapture(str(video_path))
            raw_saved = 0
            frame_num = 0
            while raw_saved < 5:
                ret, frame = cap_raw.read()
                if not ret:
                    break
                img_path = images_dir / f"raw_frame_{frame_num:06d}.jpg"
                cv2.imwrite(str(img_path), frame)
                metadata[crow_id]["images"][str(img_path)] = {
                    "frame": frame_num,
                    "bbox": None,
                    "score": None
                }
                raw_saved += 1
                frame_num += 1
            cap_raw.release()

        # Extract audio using ffmpeg
        audio_path = audio_dir / f"{video_path.stem}.wav"
        try:
            print("[DEBUG] video_test_data: extracting audio from", video_path, "to", audio_path)
            subprocess.run([
                "ffmpeg", "-i", str(video_path),
                "-vn",  # No video
                "-ar", "44100",  # Sample rate
                "-ac", "1",  # Mono
                "-y",  # Overwrite output file if it exists
                str(audio_path)
            ], check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            print("[DEBUG] video_test_data: ffmpeg extraction failed for", video_path, "stderr:", e.stderr.decode())
            continue

    # Save metadata file
    metadata_path = base_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return {
        "base_dir": base_dir,
        "valid": valid_videos_found,
        "metadata": metadata
    }

@pytest.fixture(scope="session")
def video_dataset(video_test_data, request):
    """Create a dataset from video test data."""
    from dataset import CrowTripletDataset
    try:
        # (The fixture now creates a minimal CrowTripletDataset–compatible structure in video_test_data['base_dir'])
        dataset = CrowTripletDataset(video_test_data['base_dir'], transform=None)
        return dataset
    except ValueError as e:
        pytest.skip(str(e)) 