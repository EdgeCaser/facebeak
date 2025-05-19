import pytest
import numpy as np
import os
import soundfile as sf
from audio import extract_audio_features
import torch

def test_audio_feature_extraction(test_data_dir):
    """Test audio feature extraction from a real audio file."""
    # Get a test audio file
    audio_path = os.path.join(test_data_dir, "crow_audio", "crow1", "audio_0.wav")
    
    # Extract features
    features = extract_audio_features(audio_path)
    
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

def test_audio_feature_extraction_parameters(test_data_dir):
    """Test audio feature extraction with different parameters."""
    audio_path = os.path.join(test_data_dir, "crow_audio", "crow1", "audio_0.wav")
    
    # Test with different sample rates
    features_16k = extract_audio_features(audio_path, sr=16000)
    features_22k = extract_audio_features(audio_path, sr=22050)
    
    assert features_16k[0].shape != features_22k[0].shape  # Different time dimensions
    
    # Test with different hop lengths
    features_hop512 = extract_audio_features(audio_path, hop_length=512)
    features_hop1024 = extract_audio_features(audio_path, hop_length=1024)
    
    assert features_hop512[0].shape[1] > features_hop1024[0].shape[1]  # More frames with smaller hop length
    
    # Test with different number of mel bands
    features_64mels = extract_audio_features(audio_path, n_mels=64)
    features_128mels = extract_audio_features(audio_path, n_mels=128)
    
    assert features_64mels[0].shape[0] == 64
    assert features_128mels[0].shape[0] == 128

def test_audio_feature_extraction_error_handling():
    """Test error handling in audio feature extraction."""
    # Test with non-existent file
    with pytest.raises(FileNotFoundError):
        extract_audio_features("non_existent.wav")
    
    # Test with invalid audio file
    invalid_path = "invalid.wav"
    with open(invalid_path, "wb") as f:
        f.write(b"invalid audio data")
    
    try:
        with pytest.raises(Exception):
            extract_audio_features(invalid_path)
    finally:
        os.remove(invalid_path)

def test_audio_feature_normalization(test_data_dir):
    """Test audio feature normalization."""
    audio_path = os.path.join(test_data_dir, "crow_audio", "crow1", "audio_0.wav")
    
    # Extract features
    mel_spec, chroma = extract_audio_features(audio_path)
    
    # Check mel spectrogram normalization
    assert mel_spec.min() >= 0  # Should be non-negative
    assert mel_spec.max() <= 1  # Should be normalized to [0, 1]
    
    # Check chroma normalization
    assert chroma.min() >= 0  # Should be non-negative
    assert chroma.max() <= 1  # Should be normalized to [0, 1]

def test_audio_feature_consistency(test_data_dir):
    """Test consistency of audio feature extraction."""
    audio_path = os.path.join(test_data_dir, "crow_audio", "crow1", "audio_0.wav")
    
    # Extract features multiple times
    features1 = extract_audio_features(audio_path)
    features2 = extract_audio_features(audio_path)
    
    # Check that features are consistent
    assert np.allclose(features1[0], features2[0])  # Mel spectrograms should be identical
    assert np.allclose(features1[1], features2[1])  # Chroma features should be identical

def test_audio_feature_tensor_conversion(test_data_dir):
    """Test conversion of audio features to tensors."""
    audio_path = os.path.join(test_data_dir, "crow_audio", "crow1", "audio_0.wav")
    
    # Extract features
    mel_spec, chroma = extract_audio_features(audio_path)
    
    # Convert to tensors
    mel_spec_tensor = torch.from_numpy(mel_spec)
    chroma_tensor = torch.from_numpy(chroma)
    
    # Check tensor properties
    assert isinstance(mel_spec_tensor, torch.Tensor)
    assert isinstance(chroma_tensor, torch.Tensor)
    assert mel_spec_tensor.shape == mel_spec.shape
    assert chroma_tensor.shape == chroma.shape
    assert mel_spec_tensor.dtype == torch.float32
    assert chroma_tensor.dtype == torch.float32 