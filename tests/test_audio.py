import pytest
import numpy as np
import os
import sys
import cv2
import subprocess
from pathlib import Path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from audio import extract_audio_features

@pytest.mark.video
def test_audio_extraction_from_video(video_test_data):
    """Test audio extraction from video files."""
    if not video_test_data['valid']:
        pytest.skip("No valid video files found in test data")
        
    base_dir = video_test_data['base_dir']
    metadata = video_test_data['metadata']
    
    # Check each crow's directory
    for crow_id, crow_data in metadata.items():
        crow_dir = base_dir / crow_id
        
        # Get video file
        video_files = list(crow_dir.glob("*.mp4"))
        if not video_files:
            continue
            
        video_path = video_files[0]
        
        # Extract audio using ffmpeg
        audio_path = crow_dir / "audio" / f"{video_path.stem}.wav"
        audio_path.parent.mkdir(exist_ok=True)
        
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
            
            # Extract features from the extracted audio
            features = extract_audio_features(str(audio_path))
            
            # Check feature types and shapes
            assert isinstance(features, tuple)
            assert len(features) == 2  # mel_spec and chroma
            
            mel_spec, chroma = features
            assert isinstance(mel_spec, np.ndarray)
            assert isinstance(chroma, np.ndarray)
            
            # Check mel spectrogram shape and properties
            assert mel_spec.ndim == 2
            assert mel_spec.shape[0] == 128  # n_mels
            assert mel_spec.shape[1] > 0  # time dimension
            assert not np.isnan(mel_spec).any()
            assert not np.isinf(mel_spec).any()
            
            # Check chroma shape and properties
            assert chroma.ndim == 2
            assert chroma.shape[0] == 12  # chroma bins
            assert chroma.shape[1] > 0  # time dimension
            assert not np.isnan(chroma).any()
            assert not np.isinf(chroma).any()
            
        except subprocess.CalledProcessError as e:
            pytest.skip(f"Failed to extract audio from {video_path}: {e.stderr.decode()}")
        finally:
            # Clean up extracted audio file
            if audio_path.exists():
                audio_path.unlink()

@pytest.mark.video
def test_audio_feature_consistency(video_test_data):
    """Test consistency of audio feature extraction from video."""
    if not video_test_data['valid']:
        pytest.skip("No valid video files found in test data")
        
    base_dir = video_test_data['base_dir']
    metadata = video_test_data['metadata']
    
    # Check each crow's directory
    for crow_id, crow_data in metadata.items():
        crow_dir = base_dir / crow_id
        
        # Get video file
        video_files = list(crow_dir.glob("*.mp4"))
        if not video_files:
            continue
            
        video_path = video_files[0]
        audio_path = crow_dir / "audio" / f"{video_path.stem}.wav"
        audio_path.parent.mkdir(exist_ok=True)
        
        try:
            # Extract audio
            subprocess.run([
                "ffmpeg", "-i", str(video_path),
                "-vn", "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "1", "-y",
                str(audio_path)
            ], check=True, capture_output=True)
            
            # Extract features multiple times
            features1 = extract_audio_features(str(audio_path))
            features2 = extract_audio_features(str(audio_path))
            
            # Check that features are consistent
            assert np.allclose(features1[0], features2[0])  # Mel spectrograms should be identical
            assert np.allclose(features1[1], features2[1])  # Chroma features should be identical
            
        except subprocess.CalledProcessError as e:
            pytest.skip(f"Failed to extract audio from {video_path}: {e.stderr.decode()}")
        finally:
            # Clean up extracted audio file
            if audio_path.exists():
                audio_path.unlink()

if __name__ == '__main__':
    pytest.main([__file__]) 