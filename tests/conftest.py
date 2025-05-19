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
def video_test_data(tmp_path_factory, request):
    """Create test data from video files. Only used by tests marked with @pytest.mark.video."""
    # Skip if test is not marked with video
    if not request.node.get_closest_marker('video'):
        pytest.skip("Test not marked as requiring video data")
    
    # Create a temporary directory for extracted data
    base = tmp_path_factory.mktemp("video_test_data")
    
    # Path to test videos
    video_dir = Path("unit testing videos")
    video_files = list(video_dir.glob("*.mp4"))
    
    if not video_files:
        pytest.skip("No test videos found in 'unit testing videos' directory")
    
    # Create directories for extracted data
    frames_dir = base / "frames"
    audio_dir = base / "audio"
    frames_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each video
    video_data = {}
    for video_path in video_files:
        video_name = video_path.stem
        video_data[video_name] = {
            'frames': [],
            'audio': None
        }
        
        # Extract frames
        cap = cv2.VideoCapture(str(video_path))
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Save every 10th frame to avoid too many files
            if frame_count % 10 == 0:
                frame_path = frames_dir / f"{video_name}_frame_{frame_count:04d}.jpg"
                cv2.imwrite(str(frame_path), frame)
                video_data[video_name]['frames'].append(str(frame_path))
            
            frame_count += 1
        cap.release()
        
        # Extract audio
        try:
            # Extract audio using librosa
            y, sr = librosa.load(str(video_path), sr=None)
            audio_path = audio_dir / f"{video_name}.wav"
            sf.write(str(audio_path), y, sr)
            video_data[video_name]['audio'] = str(audio_path)
        except Exception as e:
            logger.warning(f"Failed to extract audio from {video_path}: {e}")
    
    return {
        'base_dir': str(base),
        'frames_dir': str(frames_dir),
        'audio_dir': str(audio_dir),
        'video_data': video_data
    }

@pytest.fixture(scope="session")
def video_dataset(video_test_data, request):
    """Create a dataset from video test data. Only used by tests marked with @pytest.mark.video."""
    # Skip if test is not marked with video
    if not request.node.get_closest_marker('video'):
        pytest.skip("Test not marked as requiring video data")
    
    from dataset import CrowTripletDataset
    
    try:
        dataset = CrowTripletDataset(
            video_test_data['base_dir'],
            transform=None  # No transforms for testing
        )
        return dataset
    except ValueError as e:
        pytest.skip(str(e)) 