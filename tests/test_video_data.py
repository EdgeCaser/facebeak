import pytest
import torch
import numpy as np
import cv2
from PIL import Image
import soundfile as sf
from pathlib import Path
from audio import extract_audio_features
from model import CrowMultiModalEmbedder
from training import compute_triplet_loss, compute_metrics

@pytest.mark.video
def test_video_frame_extraction(video_test_data):
    """Integration test: Verify frame extraction from real video files."""
    base_dir = video_test_data['base_dir']
    
    # Check each crow's directory
    for crow_dir in base_dir.iterdir():
        if not crow_dir.is_dir():
            continue
            
        # Get all frame files
        frame_files = list((crow_dir / "images").glob("*.jpg"))
        assert len(frame_files) > 0, f"No frames extracted for crow {crow_dir.name}"
        
        # Check frame format
        frame_path = frame_files[0]
        img = cv2.imread(str(frame_path))
        assert img is not None, f"Failed to read frame {frame_path}"
        assert img.ndim == 3, f"Frame {frame_path} is not a color image"
        assert img.shape[2] == 3, f"Frame {frame_path} does not have 3 color channels"

@pytest.mark.video
def test_video_audio_extraction(video_test_data):
    """Integration test: Verify audio extraction from real video files."""
    base_dir = video_test_data['base_dir']
    
    # Check each crow's directory
    for crow_dir in base_dir.iterdir():
        if not crow_dir.is_dir():
            continue
            
        # Get all audio files
        audio_files = list((crow_dir / "audio").glob("*.wav"))
        assert len(audio_files) > 0, f"No audio files extracted for crow {crow_dir.name}"
        
        # Check audio file
        audio_path = audio_files[0]
        assert audio_path.exists(), f"Audio file {audio_path} does not exist"
        
        # Try to read audio file
        y, sr = sf.read(str(audio_path))
        assert len(y) > 0, f"Audio file {audio_path} is empty"
        assert sr > 0, f"Invalid sample rate in {audio_path}"

@pytest.mark.video
def test_audio_feature_extraction_video(video_test_data):
    """Integration test: Verify audio feature extraction on real video audio."""
    base_dir = video_test_data['base_dir']
    
    # Check each crow's directory
    for crow_dir in base_dir.iterdir():
        if not crow_dir.is_dir():
            continue
            
        # Get all audio files
        audio_files = list((crow_dir / "audio").glob("*.wav"))
        if not audio_files:
            continue
            
        # Extract features from first audio file
        audio_path = audio_files[0]
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

@pytest.mark.video
def test_dataset_with_video_data(video_dataset):
    """Integration test: Verify dataset functionality with real video data."""
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
    model = CrowMultiModalEmbedder(
        visual_embedding_dim=512,
        audio_embedding_dim=256,
        final_embedding_dim=128
    ).to(device)
    
    # Get a batch of data using collate_fn to pad audio
    batch_size = 2
    indices = np.random.choice(len(video_dataset), batch_size, replace=False)
    batch = video_dataset.collate_fn([video_dataset[i] for i in indices])
    batch['image'] = batch['image'].to(device)
    batch['audio']['mel_spec'] = batch['audio']['mel_spec'].to(device)
    batch['audio']['chroma'] = batch['audio']['chroma'].to(device)
    
    # Test forward pass
    embeddings = model(batch['image'], batch['audio'])
    assert isinstance(embeddings, torch.Tensor)
    assert embeddings.shape == (batch_size, model.final_embedding_dim)
    assert not torch.isnan(embeddings).any()
    assert not torch.isinf(embeddings).any()
    
    # Test triplet loss
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
    
    # Test metrics
    metrics, similarities = compute_metrics(model, batch, device)
    assert isinstance(metrics, dict)
    assert all(k in metrics for k in ['accuracy', 'precision', 'recall', 'f1'])
    assert isinstance(similarities, np.ndarray)
    assert similarities.shape == (batch_size, batch_size)
    assert np.all(similarities >= -1) and np.all(similarities <= 1) 