import pytest
import torch
import os
import sys
from pathlib import Path
import subprocess
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset import CrowTripletDataset
from torch.utils.data import DataLoader

def test_dataset_initialization(test_data_dir):
    """Test dataset initialization with test data."""
    dataset = CrowTripletDataset(str(test_data_dir))
    
    # Check that dataset loaded the correct number of samples
    assert len(dataset) > 0, "Dataset should have samples"
    
    # Check that we have samples for each crow
    crow_ids = set()
    for img_path, video_path, audio_path, crow_id in dataset.samples:
        crow_ids.add(crow_id)
    assert len(crow_ids) == 3, "Should have samples from all 3 crows"
    
    # Check that each sample has required fields
    sample = dataset[0]
    assert 'image' in sample, "Sample should have image"
    assert 'audio' in sample, "Sample should have audio"
    assert 'crow_id' in sample, "Sample should have crow_id"
    
    # Check image tensor shape
    assert sample['image'].shape == (3, 512, 512), "Image should be 3x512x512"
    
    # Check audio features if available
    if sample['audio'] is not None:
        assert 'mel_spec' in sample['audio'], "Audio should have mel spectrogram"
        assert 'chroma' in sample['audio'], "Audio should have chroma features"
        assert isinstance(sample['audio']['mel_spec'], torch.Tensor), "Mel spectrogram should be a tensor"
        assert isinstance(sample['audio']['chroma'], torch.Tensor), "Chroma should be a tensor"
        assert sample['audio']['mel_spec'].shape[0] == 128, "Mel spectrogram should have 128 bins"
        assert sample['audio']['chroma'].shape[0] == 12, "Chroma should have 12 bins"

def test_dataset_getitem(test_data_dir):
    """Test dataset __getitem__ method."""
    dataset = CrowTripletDataset(str(test_data_dir))
    
    # Test getting a single sample
    sample = dataset[0]
    
    # Check data types
    assert isinstance(sample['image'], torch.Tensor), "Image should be a tensor"
    assert isinstance(sample['crow_id'], str), "Crow ID should be a string"
    
    # Check value ranges
    assert sample['image'].min() >= -2.7 and sample['image'].max() <= 2.7, "Image values should be normalized"
    
    # Check audio if available
    if sample['audio'] is not None:
        assert isinstance(sample['audio'], dict), "Audio should be a dictionary"
        assert not torch.isnan(sample['audio']['mel_spec']).any(), "Mel spectrogram should not contain NaN values"
        assert not torch.isnan(sample['audio']['chroma']).any(), "Chroma should not contain NaN values"
    
    # Select one sample from each crow
    crow_indices = {}
    for i, (_, _, _, crow_id) in enumerate(dataset.samples):
        if crow_id not in crow_indices:
            crow_indices[crow_id] = i
    indices = list(crow_indices.values())
    samples = [dataset[i] for i in indices]
    
    # Check that samples are from different crows
    crow_ids = [s['crow_id'] for s in samples]
    assert len(set(crow_ids)) > 1, "Samples should be from different crows"

def test_dataset_dataloader(test_data_dir):
    """Test dataset with DataLoader."""
    dataset = CrowTripletDataset(str(test_data_dir))
    dataloader = DataLoader(
        dataset, 
        batch_size=4, 
        shuffle=True, 
        num_workers=0,
        collate_fn=dataset.collate_fn
    )
    
    # Test getting a batch
    batch = next(iter(dataloader))
    
    # Check batch structure
    assert 'image' in batch, "Batch should have image"
    assert 'audio' in batch, "Batch should have audio"
    assert 'crow_id' in batch, "Batch should have crow_id"
    
    # Check shapes
    assert batch['image'].shape[0] == 4, "Batch should have 4 images"
    assert batch['image'].shape[1:] == (3, 512, 512), "Images should be 3x512x512"
    
    if batch['audio'] is not None:
        assert 'mel_spec' in batch['audio'], "Batch audio should have mel spectrogram"
        assert 'chroma' in batch['audio'], "Batch audio should have chroma features"
        assert batch['audio']['mel_spec'].shape[0] == 4, "Batch should have 4 mel spectrograms"
        assert batch['audio']['chroma'].shape[0] == 4, "Batch should have 4 chroma features"
    
    assert len(batch['crow_id']) == 4, "Batch should have 4 crow IDs"

def test_dataset_transforms(test_data_dir):
    """Test dataset transforms."""
    dataset = CrowTripletDataset(str(test_data_dir))
    
    # Get two samples
    sample1 = dataset[0]
    sample2 = dataset[0]  # Same sample
    
    # Check that transforms are applied consistently
    assert torch.allclose(sample1['image'], sample2['image']), "Transforms should be deterministic"
    
    # Check that audio features are consistent
    if sample1['audio'] is not None and sample2['audio'] is not None:
        assert torch.allclose(sample1['audio']['mel_spec'], sample2['audio']['mel_spec']), "Mel spectrograms should be consistent"
        assert torch.allclose(sample1['audio']['chroma'], sample2['audio']['chroma']), "Chroma features should be consistent"

def test_dataset_temporal_consistency(test_data_dir):
    """Test temporal consistency of samples."""
    dataset = CrowTripletDataset(str(test_data_dir))
    
    # Group samples by crow
    crow_samples = {}
    for img_path, video_path, audio_path, crow_id in dataset.samples:
        if crow_id not in crow_samples:
            crow_samples[crow_id] = []
        crow_samples[crow_id].append((img_path, video_path))
    
    # Check that each crow has multiple samples
    for crow_id, samples in crow_samples.items():
        assert len(samples) > 1, f"Crow {crow_id} should have multiple samples"
        
        # Check that samples have both image and video
        for img_path, video_path in samples:
            assert img_path.exists(), f"Image file {img_path} should exist"
            assert video_path.exists(), f"Video file {video_path} should exist"

if __name__ == '__main__':
    pytest.main([__file__]) 