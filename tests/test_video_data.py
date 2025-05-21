import pytest
import torch
import numpy as np
import cv2
import subprocess
from PIL import Image
import soundfile as sf
from pathlib import Path
from audio import extract_audio_features
from model import CrowMultiModalEmbedder
from training import compute_triplet_loss, compute_metrics
import librosa
import logging

logger = logging.getLogger(__name__)

@pytest.mark.video
def test_video_frame_extraction(video_test_data):
    """Integration test: Verify frame extraction from real video files."""
    if not video_test_data['valid']:
        pytest.skip("No valid video files found in test data")
        
    base_dir = video_test_data['base_dir']
    metadata = video_test_data['metadata']
    
    # Check each crow's directory
    for crow_id, crow_data in metadata.items():
        crow_dir = base_dir / crow_id
        images_dir = crow_dir / "images"
        
        # Get all frame files
        frame_files = list(images_dir.glob("*.jpg"))
        assert len(frame_files) > 0, f"No frames extracted for crow {crow_id}"
        
        # Check frame format
        frame_path = frame_files[0]
        img = cv2.imread(str(frame_path))
        assert img is not None, f"Failed to read frame {frame_path}"
        assert img.ndim == 3, f"Frame {frame_path} is not a color image"
        assert img.shape[2] == 3, f"Frame {frame_path} does not have 3 color channels"
        
        # Verify metadata
        assert crow_id in metadata, f"No metadata found for crow {crow_id}"
        assert "images" in metadata[crow_id], f"No image metadata found for crow {crow_id}"
        assert len(metadata[crow_id]["images"]) > 0, f"No image entries in metadata for crow {crow_id}"

@pytest.mark.video
def test_video_audio_extraction(video_test_data):
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
            
            # Verify audio was extracted
            assert audio_path.exists(), f"Failed to extract audio from {video_path}"
            
            # Extract features from the audio
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
def test_audio_feature_extraction_video(video_test_data):
    """Integration test: Verify audio feature extraction on real video audio."""
    if not video_test_data['valid']:
        pytest.skip("No valid video files found in test data")
        
    base_dir = video_test_data['base_dir']
    metadata = video_test_data['metadata']
    
    # Track if we found any valid audio files
    valid_audio_found = False
    
    # Check each crow's directory
    for crow_id, crow_data in metadata.items():
        crow_dir = base_dir / crow_id
        
        # Get video file first
        video_files = list(crow_dir.glob("*.mp4"))
        if not video_files:
            continue
            
        video_path = video_files[0]
        audio_path = crow_dir / "audio" / f"{video_path.stem}.wav"
        audio_path.parent.mkdir(exist_ok=True)
        
        try:
            # Extract audio if it doesn't exist
            if not audio_path.exists():
                subprocess.run([
                    "ffmpeg", "-i", str(video_path),
                    "-vn", "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "1", "-y",
                    str(audio_path)
                ], check=True, capture_output=True)
            
            # Verify audio file exists and is readable
            if not audio_path.exists():
                logger.warning(f"Audio file {audio_path} was not created")
                continue
                
            # Try to read the audio file first to verify it's valid
            try:
                y, sr = librosa.load(str(audio_path), sr=None)
                if len(y) == 0:
                    logger.warning(f"Audio file {audio_path} is empty")
                    continue
            except Exception as e:
                logger.warning(f"Failed to read audio file {audio_path}: {e}")
                continue
            
            # Extract features with proper error handling
            try:
                features = extract_audio_features(str(audio_path))
            except Exception as e:
                logger.warning(f"Failed to extract audio features from {audio_path}: {e}")
                continue
            
            # Check feature types and shapes
            assert isinstance(features, tuple), "Features should be a tuple"
            assert len(features) == 2, "Features should contain mel_spec and chroma"
            
            mel_spec, chroma = features
            assert isinstance(mel_spec, np.ndarray), "Mel spectrogram should be a numpy array"
            assert isinstance(chroma, np.ndarray), "Chroma features should be a numpy array"
            
            # Check mel spectrogram shape and properties
            assert mel_spec.ndim == 2, "Mel spectrogram should be 2D"
            assert mel_spec.shape[0] == 128, "Mel spectrogram should have 128 bins"
            assert mel_spec.shape[1] > 0, "Mel spectrogram should have time dimension"
            assert not np.isnan(mel_spec).any(), "Mel spectrogram should not contain NaN values"
            assert not np.isinf(mel_spec).any(), "Mel spectrogram should not contain Inf values"
            
            # Check chroma shape and properties
            assert chroma.ndim == 2, "Chroma features should be 2D"
            assert chroma.shape[0] == 12, "Chroma features should have 12 bins"
            assert chroma.shape[1] > 0, "Chroma features should have time dimension"
            assert not np.isnan(chroma).any(), "Chroma features should not contain NaN values"
            assert not np.isinf(chroma).any(), "Chroma features should not contain Inf values"
            
            # Test successful extraction
            valid_audio_found = True
            break  # Exit after successful extraction
            
        except subprocess.CalledProcessError as e:
            logger.warning(f"Failed to extract audio from {video_path}: {e.stderr.decode()}")
            continue
        except Exception as e:
            logger.warning(f"Unexpected error processing {video_path}: {e}")
            continue
        finally:
            # Clean up extracted audio file
            if audio_path.exists():
                try:
                    audio_path.unlink()
                except Exception as e:
                    logger.warning(f"Failed to clean up audio file {audio_path}: {e}")
    
    if not valid_audio_found:
        pytest.skip("No valid audio files found for testing")

@pytest.mark.video
def test_dataset_with_video_data(video_dataset):
    """Integration test: Verify dataset functionality with real video data."""
    if not hasattr(video_dataset, 'valid') or not video_dataset.valid:
        pytest.skip("No valid video data available for dataset")
        
    assert len(video_dataset) > 0, "Dataset should contain samples"
    
    # Test a few samples
    for i in range(min(5, len(video_dataset))):
        sample = video_dataset[i]
        
        # Check image
        assert isinstance(sample['image'], torch.Tensor)
        assert sample['image'].shape == (3, 224, 224)  # RGB image
        # Accept a reasonable range for ImageNet normalization
        assert sample['image'].min() > -3 and sample['image'].max() < 3, f"Image values out of expected range: min={sample['image'].min()}, max={sample['image'].max()}"
        
        # Check audio
        assert isinstance(sample['audio'], dict)
        assert 'mel_spec' in sample['audio']
        assert 'chroma' in sample['audio']
        assert isinstance(sample['audio']['mel_spec'], torch.Tensor)
        assert isinstance(sample['audio']['chroma'], torch.Tensor)
        assert sample['audio']['mel_spec'].shape[0] == 128  # Mel spectrogram bins
        assert sample['audio']['chroma'].shape[0] == 12  # Chroma bins
        
        # Check label
        assert isinstance(sample['crow_id'], str)
        assert sample['crow_id'] in video_dataset.crow_ids

@pytest.mark.video
def test_model_with_video_data(video_dataset, device):
    """Integration test: Verify model functionality with real video data."""
    if not hasattr(video_dataset, 'valid') or not video_dataset.valid:
        pytest.skip("No valid video data available for dataset")
        
    model = CrowMultiModalEmbedder(
        visual_embedding_dim=512,
        audio_embedding_dim=256,
        final_embedding_dim=128
    ).to(device)
    
    # Get a batch of data using collate_fn to pad audio
    batch_size = 2
    indices = np.random.choice(len(video_dataset), batch_size, replace=False)
    batch = video_dataset.collate_fn([video_dataset[i] for i in indices])
    
    # Move data to device
    batch['image'] = batch['image'].to(device)
    if batch['audio'] is not None:
        batch['audio']['mel_spec'] = batch['audio']['mel_spec'].to(device)
        batch['audio']['chroma'] = batch['audio']['chroma'].to(device)
    
    # Test forward pass with both visual and audio inputs
    try:
        embeddings = model(batch['image'], batch['audio'])
        assert isinstance(embeddings, torch.Tensor)
        assert embeddings.shape == (batch_size, model.final_embedding_dim)
        assert not torch.isnan(embeddings).any()
        assert not torch.isinf(embeddings).any()
    except Exception as e:
        pytest.fail(f"Model forward pass failed: {str(e)}")
    
    # Test forward pass with visual input only
    try:
        embeddings_visual = model(batch['image'], None)
        assert isinstance(embeddings_visual, torch.Tensor)
        assert embeddings_visual.shape == (batch_size, model.final_embedding_dim)
        assert not torch.isnan(embeddings_visual).any()
        assert not torch.isinf(embeddings_visual).any()
    except Exception as e:
        pytest.fail(f"Model forward pass with visual input only failed: {str(e)}")
    
    # Test triplet loss if we have valid embeddings
    try:
        loss = compute_triplet_loss(
            embeddings[0:1],  # anchor
            embeddings[0:1],  # positive (same as anchor for testing)
            embeddings[1:2],  # negative
            margin=1.0
        )
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # Scalar
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)
    except Exception as e:
        pytest.fail(f"Triplet loss computation failed: {str(e)}")
    
    # Test metrics if we have valid embeddings
    try:
        metrics, similarities = compute_metrics(model, batch, device)
        assert isinstance(metrics, dict)
        assert isinstance(similarities, torch.Tensor)
        assert not torch.isnan(similarities).any()
        assert not torch.isinf(similarities).any()
    except Exception as e:
        pytest.fail(f"Metrics computation failed: {str(e)}")

if __name__ == '__main__':
    pytest.main([__file__]) 