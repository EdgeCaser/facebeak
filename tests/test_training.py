import pytest
import torch
import numpy as np
from training import compute_triplet_loss, compute_metrics, training_step, validation_step
from dataset import CrowTripletDataset
from model import CrowMultiModalEmbedder

def test_triplet_loss():
    """Test triplet loss computation."""
    # Create dummy embeddings where anchor-positive are close, anchor-negative are far
    anchor = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float32)
    positive = torch.tensor([[0.1, -0.1], [1.1, 0.9]], dtype=torch.float32)
    negative = torch.tensor([[2.0, 2.0], [-2.0, -2.0]], dtype=torch.float32)
    
    # Compute loss
    loss = compute_triplet_loss(anchor, positive, negative, margin=1.0)
    
    # Check loss
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0  # Scalar
    assert loss >= 0  # Should be non-negative
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)
    
    # Test with identical embeddings (should give zero loss)
    loss_zero = compute_triplet_loss(anchor, anchor, negative, margin=1.0)
    assert torch.abs(loss_zero) < 1e-6  # Use small epsilon for comparison
    
    # Test with very different embeddings
    # Create a case where the negative is closer than the margin
    close_negative = torch.tensor([[0.5, 0.5], [-0.5, -0.5]], dtype=torch.float32)
    loss_close = compute_triplet_loss(anchor, positive, close_negative, margin=1.0)
    
    # Create a case where the negative is exactly at the margin
    margin_negative = torch.tensor([[1.0, 1.0], [-1.0, -1.0]], dtype=torch.float32)
    loss_margin = compute_triplet_loss(anchor, positive, margin_negative, margin=1.0)
    
    # The loss should be larger when the negative is closer than the margin
    assert loss_close > loss_margin, "Loss should be larger when negative is closer than margin"
    
    # Test with very far negative (should give zero loss as it satisfies margin)
    far_negative = torch.tensor([[10.0, 10.0], [-10.0, -10.0]], dtype=torch.float32)
    loss_far = compute_triplet_loss(anchor, positive, far_negative, margin=1.0)
    assert torch.abs(loss_far) < 1e-6, "Loss should be zero when negative is far enough"

def test_triplet_loss_edge_cases():
    """Test triplet loss with edge cases."""
    # Create dummy embeddings
    batch_size = 4
    embed_dim = 512
    anchor = torch.randn(batch_size, embed_dim)
    positive = torch.randn(batch_size, embed_dim)
    negative = torch.randn(batch_size, embed_dim)
    
    # Test with zero margin
    loss = compute_triplet_loss(anchor, positive, negative, margin=0.0)
    assert loss >= 0
    
    # Test with very large margin
    loss = compute_triplet_loss(anchor, positive, negative, margin=100.0)
    assert loss >= 0
    
    # Test with identical anchor and positive
    loss = compute_triplet_loss(anchor, anchor, negative)
    assert loss >= 0
    
    # Test with identical anchor and negative
    loss = compute_triplet_loss(anchor, positive, anchor)
    assert loss >= 0

def test_compute_metrics():
    """Test metric computation."""
    model = CrowMultiModalEmbedder(
        visual_embedding_dim=512,
        audio_embedding_dim=256,
        final_embedding_dim=128
    )
    device = torch.device('cpu')
    
    class DummyDataset:
        def __init__(self):
            self.data = []
            # Create 4 samples with 2 unique crow IDs
            for i in range(4):
                self.data.append({
                    'image': torch.randn(3, 224, 224),
                    'audio': {
                        'mel_spec': torch.randn(1, 128, 64),
                        'chroma': torch.randn(1, 12, 64)
                    },
                    'crow_id': f'crow_{i % 2}'
                })
        def __len__(self):
            return len(self.data)
        def __getitem__(self, idx):
            return self.data[idx]
    
    dataset = DummyDataset()
    # Collate samples into batch
    images = torch.stack([s['image'] for s in dataset.data])  # (4, 3, 224, 224)
    mel_specs = torch.cat([s['audio']['mel_spec'] for s in dataset.data], dim=0)  # (4, 128, 64)
    chromas = torch.cat([s['audio']['chroma'] for s in dataset.data], dim=0)  # (4, 12, 64)
    crow_ids = [s['crow_id'] for s in dataset.data]
    batch = {
        'image': images,
        'audio': {
            'mel_spec': mel_specs,
            'chroma': chromas
        },
        'crow_id': crow_ids
    }
    metrics, similarities = compute_metrics(model, batch, device)
    
    # Verify metrics structure and values
    assert isinstance(metrics, dict)
    assert all(k in metrics for k in ['accuracy', 'precision', 'recall', 'f1'])
    assert all(0 <= v <= 1 for v in metrics.values())
    
    # Verify similarities matrix
    assert isinstance(similarities, np.ndarray)
    assert similarities.shape == (4, 4)  # Should match dataset size
    assert np.all(similarities >= -1) and np.all(similarities <= 1)
    # Do not assert same-crow similarities > different-crow similarities for random model

def test_compute_metrics_edge_cases():
    """Test compute_metrics with edge cases."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CrowMultiModalEmbedder(
        visual_embedding_dim=512,
        audio_embedding_dim=256,
        final_embedding_dim=128
    ).to(device)
    
    # Test with empty dataset
    empty_dataset = {
        'image': torch.empty(0, 3, 224, 224, device=device),
        'audio': {
            'mel_spec': torch.empty(0, 128, 64, device=device),
            'chroma': torch.empty(0, 12, 64, device=device)
        },
        'crow_id': []
    }
    metrics, similarities = compute_metrics(model, empty_dataset, device)
    assert isinstance(metrics, dict)
    assert all(k in metrics for k in ['accuracy', 'precision', 'recall', 'f1'])
    assert all(v == 0.0 for v in metrics.values())
    assert similarities.size == 0
    
    # Test with single sample
    single_sample = {
        'image': torch.randn(1, 3, 224, 224, device=device),
        'audio': {
            'mel_spec': torch.randn(1, 128, 64, device=device),
            'chroma': torch.randn(1, 12, 64, device=device)
        },
        'crow_id': ['crow1']
    }
    metrics, similarities = compute_metrics(model, single_sample, device)
    assert isinstance(metrics, dict)
    assert all(k in metrics for k in ['accuracy', 'precision', 'recall', 'f1'])
    assert similarities.shape == (1, 1)
    assert np.all(similarities >= -1) and np.all(similarities <= 1)
    
    # Test with identical samples
    identical_samples = {
        'image': torch.randn(2, 3, 224, 224, device=device),
        'audio': {
            'mel_spec': torch.randn(2, 128, 64, device=device),
            'chroma': torch.randn(2, 12, 64, device=device)
        },
        'crow_id': ['crow1', 'crow1']
    }
    metrics, similarities = compute_metrics(model, identical_samples, device)
    assert isinstance(metrics, dict)
    assert all(k in metrics for k in ['accuracy', 'precision', 'recall', 'f1'])
    assert similarities.shape == (2, 2)
    assert np.all(similarities >= -1) and np.all(similarities <= 1)
    # Similarity between identical samples should be close to 1, with a more lenient tolerance
    assert np.allclose(similarities[0, 1], 1.0, atol=5e-3), \
        f"Similarity between identical samples should be close to 1, got {similarities[0, 1]}"

def test_training_step():
    """Test training step."""
    # Create model and optimizer
    model = CrowMultiModalEmbedder(
        visual_embedding_dim=512,
        audio_embedding_dim=256,
        final_embedding_dim=128
    )
    optimizer = torch.optim.Adam(model.parameters())
    device = torch.device('cpu')
    
    # Create dummy batch
    batch = {
        'image': torch.randn(2, 3, 224, 224),
        'audio': {
            'mel_spec': torch.randn(2, 128, 64),
            'chroma': torch.randn(2, 12, 64)
        },
        'crow_id': ['crow_1', 'crow_2']
    }
    
    # Run training step
    loss, metrics = training_step(model, optimizer, batch, device)
    
    # Check loss
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0  # Scalar
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)
    
    # Check metrics
    assert isinstance(metrics, dict)
    assert all(isinstance(v, float) for v in metrics.values())

def test_validation_step():
    """Test validation step."""
    # Create model
    model = CrowMultiModalEmbedder(
        visual_embedding_dim=512,
        audio_embedding_dim=256,
        final_embedding_dim=128
    )
    device = torch.device('cpu')
    
    # Create dummy batch
    batch = {
        'image': torch.randn(2, 3, 224, 224),
        'audio': {
            'mel_spec': torch.randn(2, 128, 64),
            'chroma': torch.randn(2, 12, 64)
        },
        'crow_id': ['crow_1', 'crow_2']
    }
    
    # Run validation step
    loss, metrics = validation_step(model, batch, device)
    
    # Check loss
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0  # Scalar
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)
    
    # Check metrics
    assert isinstance(metrics, dict)
    assert all(isinstance(v, float) for v in metrics.values()) 