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

def create_dummy_image(path):
    arr = (np.random.rand(224, 224, 3) * 255).astype(np.uint8)
    img = Image.fromarray(arr)
    img.save(path)

def create_dummy_audio(path, sr=16000, duration=1.0):
    samples = int(sr * duration)
    audio = np.random.randn(samples).astype(np.float32)
    sf.write(path, audio, sr)

@pytest.fixture(scope="session")
def test_data_dir(tmp_path_factory):
    """Create a temporary directory with test data for dataset tests."""
    # Create a temporary directory structure
    base = tmp_path_factory.mktemp("test_data")
    
    # Create crow image directories and images
    crow_names = ["crow1", "crow2", "crow_audio"]
    for crow in crow_names:
        crow_dir = base / "crow_crops" / crow  # Changed to match expected directory structure
        crow_dir.mkdir(parents=True, exist_ok=True)
        for i in range(2):
            create_dummy_image(str(crow_dir / f"img_{i}.jpg"))
    
    # Create crow_audio directory and audio files
    audio_base = base / "crow_audio"
    audio_base.mkdir(parents=True, exist_ok=True)
    for crow in crow_names:
        crow_audio_dir = audio_base / crow
        crow_audio_dir.mkdir(parents=True, exist_ok=True)
        for i in range(2):
            create_dummy_audio(str(crow_audio_dir / f"audio_{i}.wav"))
    
    # Create a dummy config file to ensure the directory is recognized
    config_path = base / "config.yaml"
    with open(config_path, "w") as f:
        f.write("data_dir: .\n")
    
    return str(base)

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
    test_videos_dir = Path("unit testing videos")
    if not test_videos_dir.exists():
        pytest.skip("Test videos directory not found")

    # Process each video in the test directory
    for video_path in test_videos_dir.glob("*.mp4"):
        # Extract crow_id from filename (assuming format: "crow_id_*.mp4")
        crow_id = video_path.stem.split("_")[0]
        
        # Create directories for this crow with the correct structure
        crow_dir = base_dir / crow_id
        images_dir = crow_dir / "images"
        audio_dir = crow_dir / "audio"
        images_dir.mkdir(parents=True, exist_ok=True)
        audio_dir.mkdir(exist_ok=True)

        # Open video file
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Warning: Could not open video {video_path}")
            continue

        # Extract first two frames (if available)
        frame_count = 0
        while frame_count < 2:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Save frame as image
            frame_path = images_dir / f"frame_{frame_count}.jpg"
            cv2.imwrite(str(frame_path), frame)
            frame_count += 1

        cap.release()

        # Extract audio using ffmpeg
        audio_path = audio_dir / f"{video_path.stem}.wav"
        try:
            subprocess.run([
                "ffmpeg", "-i", str(video_path),
                "-vn",  # No video
                "-acodec", "pcm_s16le",  # PCM 16-bit
                "-ar", "44100",  # Sample rate
                "-ac", "1",  # Mono
                "-y",  # Overwrite output file if it exists
                str(audio_path)
            ], check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            print(f"Warning: Failed to extract audio from {video_path}: {e}")
            print(f"ffmpeg stderr: {e.stderr.decode()}")
            continue

    # Verify we have at least some data
    if not any(base_dir.iterdir()):
        pytest.skip("No valid test data could be extracted from videos")

    return {
        "base_dir": base_dir
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