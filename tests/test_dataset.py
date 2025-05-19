import pytest
import torch
import os
import numpy as np
from PIL import Image
import torchvision.transforms as T
from dataset import CrowTripletDataset
from torch.utils.data import DataLoader

def test_dataset_initialization(test_data_dir):
    """Test dataset initialization with both image and audio data."""
    try:
        dataset = CrowTripletDataset(test_data_dir)
    except ValueError as e:
        pytest.skip(str(e))
    assert len(dataset) > 0
    assert isinstance(dataset.crow_to_audio, dict)

def test_dataset_getitem(test_data_dir):
    """Test dataset item retrieval."""
    try:
        dataset = CrowTripletDataset(test_data_dir)
    except ValueError as e:
        pytest.skip(str(e))
    sample = dataset[0]
    # Check image
    assert isinstance(sample['image'], torch.Tensor)
    assert sample['image'].shape == (3, 224, 224)  # RGB image
    assert torch.all(sample['image'] >= -1) and torch.all(sample['image'] <= 1)  # Normalized to [-1, 1]
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
    assert sample['crow_id'] in dataset.crow_ids

def test_dataset_dataloader(test_data_dir):
    """Test dataset with DataLoader."""
    try:
        dataset = CrowTripletDataset(test_data_dir)
    except ValueError as e:
        pytest.skip(str(e))
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        collate_fn=dataset.collate_fn
    )
    batch = next(iter(dataloader))
    # Check batch structure
    assert isinstance(batch, dict)
    assert 'image' in batch
    assert 'audio' in batch
    assert 'crow_id' in batch
    # Check image batch
    assert isinstance(batch['image'], torch.Tensor)
    assert batch['image'].shape == (2, 3, 224, 224)
    # Check audio batch
    assert isinstance(batch['audio'], dict)
    assert 'mel_spec' in batch['audio']
    assert 'chroma' in batch['audio']
    assert isinstance(batch['audio']['mel_spec'], torch.Tensor)
    assert isinstance(batch['audio']['chroma'], torch.Tensor)
    assert batch['audio']['mel_spec'].shape[0] == 2  # Batch size
    assert batch['audio']['chroma'].shape[0] == 2  # Batch size
    # Check labels
    assert isinstance(batch['crow_id'], list)
    assert len(batch['crow_id']) == 2
    assert all(isinstance(label, str) for label in batch['crow_id'])

def test_dataset_transforms(test_data_dir):
    """Test dataset transforms."""
    try:
        dataset = CrowTripletDataset(
            test_data_dir,
            transform=T.Compose([
                T.RandomHorizontalFlip(),
                T.RandomRotation(10),
                T.ColorJitter(brightness=0.2, contrast=0.2)
            ])
        )
    except ValueError as e:
        pytest.skip(str(e))
    sample = dataset[0]
    # Check image
    assert isinstance(sample['image'], torch.Tensor)
    assert sample['image'].shape == (3, 224, 224)
    assert torch.all(sample['image'] >= -1) and torch.all(sample['image'] <= 1)
    # Check audio (should be unchanged)
    assert isinstance(sample['audio'], dict)
    assert 'mel_spec' in sample['audio']
    assert 'chroma' in sample['audio']
    assert isinstance(sample['audio']['mel_spec'], torch.Tensor)
    assert isinstance(sample['audio']['chroma'], torch.Tensor)
    assert sample['audio']['mel_spec'].shape[0] == 128
    assert sample['audio']['chroma'].shape[0] == 12

def test_dataset_error_handling(test_data_dir):
    """Test dataset error handling."""
    # Test with non-existent directory
    with pytest.raises(FileNotFoundError):
        CrowTripletDataset(os.path.join(test_data_dir, "non_existent_dir"))
    
    # Test with empty directory
    empty_dir = os.path.join(test_data_dir, "empty_dir")
    os.makedirs(empty_dir, exist_ok=True)
    with pytest.raises(ValueError):
        CrowTripletDataset(empty_dir)
    
    # Test with directory containing no images
    no_images_dir = os.path.join(test_data_dir, "no_images")
    os.makedirs(no_images_dir, exist_ok=True)
    with open(os.path.join(no_images_dir, "dummy.txt"), "w") as f:
        f.write("dummy")
    with pytest.raises(ValueError):
        CrowTripletDataset(no_images_dir) 