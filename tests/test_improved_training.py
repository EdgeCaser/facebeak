#!/usr/bin/env python3
"""
Comprehensive tests for the improved training system.
Tests ImprovedTrainer, ImprovedTripletLoss, FocalTripletLoss, and all integrations.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import tempfile
import json
import os
from pathlib import Path
import shutil
from unittest.mock import Mock, patch, MagicMock, call
import logging

# Import modules to test
from improved_triplet_loss import (
    ImprovedTripletLoss, 
    FocalTripletLoss, 
    create_margin_schedule
)

# We'll need to mock some imports that may not exist yet
import sys


class MockImprovedCrowTripletDataset:
    """Mock dataset for testing."""
    def __init__(self, crop_dir, split='train', transform_mode='standard'):
        self.crop_dir = crop_dir
        self.split = split
        self.transform_mode = transform_mode
        self.crow_to_imgs = {'crow1': ['img1.jpg', 'img2.jpg'], 'crow2': ['img3.jpg', 'img4.jpg']}
        
    def __len__(self):
        return 20  # Mock dataset size
        
    def __getitem__(self, idx):
        return (
            (torch.randn(3, 224, 224), torch.randn(3, 224, 224), torch.randn(3, 224, 224)),  # anchor, pos, neg images
            (torch.randn(128, 64), torch.randn(128, 64), torch.randn(128, 64)),  # anchor, pos, neg audio
            torch.randint(0, 5, (1,))[0]  # labels
        )
        
    def update_curriculum(self, epoch):
        pass


class MockCrowResNetEmbedder(nn.Module):
    """Mock model for testing."""
    def __init__(self, embedding_dim=512):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.linear = nn.Linear(3*224*224, embedding_dim)  # Simple mock
        
    def forward(self, x):
        batch_size = x.shape[0]
        x_flat = x.view(batch_size, -1)
        return self.linear(x_flat)


# Mock the imports that train_improved.py needs
class MockImports:
    def __init__(self):
        self.CrowResNetEmbedder = MockCrowResNetEmbedder
        self.ImprovedCrowTripletDataset = MockImprovedCrowTripletDataset


@pytest.fixture
def mock_imports():
    """Mock the imports for train_improved.py"""
    mock_models = MockImports()
    mock_dataset = MockImports()
    
    with patch.dict('sys.modules', {
        'models': mock_models,
        'improved_dataset': mock_dataset
    }):
        yield mock_models


@pytest.fixture
def temp_training_dir():
    """Create temporary directory for training tests."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create mock crop directory structure
        crop_dir = Path(temp_dir) / "crow_crops"
        crop_dir.mkdir()
        
        for crow_id in ["crow1", "crow2", "crow3"]:
            crow_path = crop_dir / crow_id
            crow_path.mkdir()
            for i in range(3):
                img_file = crow_path / f"img_{i}.jpg"
                img_file.touch()
        
        yield temp_dir


@pytest.fixture
def basic_config():
    """Basic training configuration for testing."""
    return {
        'crop_dir': 'test_crops',
        'embedding_dim': 256,
        'epochs': 10,
        'batch_size': 8,
        'learning_rate': 0.001,
        'margin': 1.0,
        'mining_type': 'adaptive',
        'output_dir': 'test_output',
        'eval_every': 2,
        'save_every': 5,
        'weight_decay': 1e-4,
        'alpha': 0.2,
        'beta': 0.02,
        'num_workers': 0,  # Avoid multiprocessing in tests
        'early_stopping': False,
        'plot_every': 5
    }


class TestImprovedTripletLoss:
    """Test suite for ImprovedTripletLoss."""
    
    def test_initialization(self):
        """Test ImprovedTripletLoss initialization."""
        loss_fn = ImprovedTripletLoss(margin=1.5, mining_type='hard', alpha=0.3, beta=0.05)
        
        assert loss_fn.margin == 1.5
        assert loss_fn.mining_type == 'hard'
        assert loss_fn.alpha == 0.3
        assert loss_fn.beta == 0.05
        assert loss_fn.epoch == 0
    
    def test_invalid_mining_type(self):
        """Test initialization with invalid mining type."""
        with pytest.raises(ValueError, match="Unknown mining type"):
            loss_fn = ImprovedTripletLoss(mining_type='invalid')
            # Trigger the error by calling forward
            embeddings = torch.randn(4, 128)
            labels = torch.tensor([0, 0, 1, 1])
            loss_fn(embeddings, labels)
    
    def test_pairwise_distances(self):
        """Test pairwise distance computation."""
        loss_fn = ImprovedTripletLoss()
        embeddings = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=torch.float32)
        
        distances = loss_fn._compute_pairwise_distances(embeddings)
        
        assert distances.shape == (3, 3)
        assert torch.allclose(distances[0, 0], torch.tensor(0.0), atol=1e-6)  # Self distance
        assert distances[0, 1] > 0  # Different points
        assert torch.allclose(distances[0, 1], distances[1, 0])  # Symmetric
    
    def test_valid_pairs(self):
        """Test valid pair generation."""
        loss_fn = ImprovedTripletLoss()
        labels = torch.tensor([0, 0, 1, 1, 2])
        
        anchor_pos, anchor_neg = loss_fn._get_valid_pairs(labels)
        
        # Check positive pairs (same class)
        assert len(anchor_pos) > 0
        for anchor, positive in anchor_pos:
            assert labels[anchor] == labels[positive]
            assert anchor != positive
        
        # Check negative pairs (different class)
        assert len(anchor_neg) > 0
        for anchor, negative in anchor_neg:
            assert labels[anchor] != labels[negative]
    
    @pytest.mark.parametrize("mining_type", ["hard", "semi_hard", "adaptive", "curriculum"])
    def test_mining_strategies(self, mining_type):
        """Test different mining strategies."""
        loss_fn = ImprovedTripletLoss(mining_type=mining_type, margin=1.0)
        
        # Create embeddings with clear cluster structure
        embeddings = torch.tensor([
            [1.0, 0.0], [1.1, 0.1],  # Class 0
            [0.0, 1.0], [0.1, 1.1],  # Class 1
            [-1.0, 0.0], [-1.1, 0.1]  # Class 2
        ], dtype=torch.float32)
        labels = torch.tensor([0, 0, 1, 1, 2, 2])
        
        loss, stats = loss_fn(embeddings, labels)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad
        assert loss.item() >= 0
        assert isinstance(stats, dict)
        assert stats['mining_type'] == mining_type
        assert 'n_triplets' in stats or f'n_{mining_type}_triplets' in stats
    
    def test_hard_mining_details(self):
        """Test hard mining strategy in detail."""
        loss_fn = ImprovedTripletLoss(mining_type='hard')
        
        # Create controlled embeddings
        embeddings = torch.tensor([
            [0.0, 0.0], [0.1, 0.0], [0.2, 0.0],  # Class 0: close together
            [2.0, 0.0], [2.1, 0.0]                # Class 1: close together, far from class 0
        ], dtype=torch.float32)
        labels = torch.tensor([0, 0, 0, 1, 1])
        
        loss, stats = loss_fn(embeddings, labels)
        
        assert stats['mining_type'] == 'hard'
        assert 'n_hard_triplets' in stats
        assert 'avg_positive_dist' in stats
        assert 'avg_negative_dist' in stats
        assert stats['avg_negative_dist'] > stats['avg_positive_dist']  # Negatives should be farther
    
    def test_adaptive_mining_quality(self):
        """Test adaptive mining embedding quality calculation."""
        loss_fn = ImprovedTripletLoss(mining_type='adaptive')
        
        # Create high-quality embeddings (well separated)
        embeddings = torch.tensor([
            [0.0, 0.0], [0.1, 0.1],  # Class 0: tight cluster
            [3.0, 3.0], [3.1, 3.1]   # Class 1: tight cluster, far from class 0
        ], dtype=torch.float32)
        labels = torch.tensor([0, 0, 1, 1])
        
        loss, stats = loss_fn(embeddings, labels)
        
        assert stats['mining_type'] == 'adaptive'
        assert 'embedding_quality' in stats
        assert 'adaptive_margin' in stats
        assert stats['embedding_quality'] > 0  # Should detect good separation
    
    def test_curriculum_mining_progression(self):
        """Test curriculum mining difficulty progression."""
        loss_fn = ImprovedTripletLoss(mining_type='curriculum')
        
        embeddings = torch.randn(8, 64)
        labels = torch.tensor([0, 0, 1, 1, 2, 2, 0, 1])
        
        # Test early epoch (easy)
        loss_fn.update_epoch(0)
        loss_early, stats_early = loss_fn(embeddings, labels)
        
        # Test late epoch (hard)
        loss_fn.update_epoch(50)
        loss_late, stats_late = loss_fn(embeddings, labels)
        
        assert stats_early['difficulty'] < stats_late['difficulty']
        assert stats_early['curriculum_margin'] > stats_late['curriculum_margin']
    
    def test_empty_batches(self):
        """Test handling of empty or single-class batches."""
        loss_fn = ImprovedTripletLoss()
        
        # Empty batch
        embeddings_empty = torch.empty(0, 128)
        labels_empty = torch.empty(0, dtype=torch.long)
        loss, stats = loss_fn(embeddings_empty, labels_empty)
        assert loss.item() == 0.0
        
        # Single class (no negatives)
        embeddings_single = torch.randn(4, 128)
        labels_single = torch.tensor([0, 0, 0, 0])
        loss, stats = loss_fn(embeddings_single, labels_single)
        assert loss.item() == 0.0
    
    def test_update_epoch(self):
        """Test epoch updating functionality."""
        loss_fn = ImprovedTripletLoss()
        
        assert loss_fn.epoch == 0
        loss_fn.update_epoch(10)
        assert loss_fn.epoch == 10
    
    def test_margin_schedule_integration(self):
        """Test margin scheduling integration."""
        schedule_fn = lambda epoch: 1.0 - 0.01 * epoch
        loss_fn = ImprovedTripletLoss(margin=1.0, margin_schedule=schedule_fn)
        
        # This is tested through the _compute_triplet_loss method indirectly
        embeddings = torch.randn(6, 64)
        labels = torch.tensor([0, 0, 1, 1, 2, 2])
        
        loss_fn.update_epoch(0)
        loss_0, _ = loss_fn(embeddings, labels)
        
        loss_fn.update_epoch(10)
        loss_10, _ = loss_fn(embeddings, labels)
        
        # Both should be valid tensors
        assert isinstance(loss_0, torch.Tensor)
        assert isinstance(loss_10, torch.Tensor)


class TestFocalTripletLoss:
    """Test suite for FocalTripletLoss."""
    
    def test_initialization(self):
        """Test FocalTripletLoss initialization."""
        focal_loss = FocalTripletLoss(margin=1.5, alpha=0.3, gamma=3.0)
        
        assert focal_loss.margin == 1.5
        assert focal_loss.alpha == 0.3
        assert focal_loss.gamma == 3.0
    
    def test_forward_pass(self):
        """Test FocalTripletLoss forward pass."""
        focal_loss = FocalTripletLoss()
        
        batch_size = 4
        embed_dim = 128
        anchor = torch.randn(batch_size, embed_dim, requires_grad=True)
        positive = torch.randn(batch_size, embed_dim, requires_grad=True)
        negative = torch.randn(batch_size, embed_dim, requires_grad=True)
        
        loss = focal_loss(anchor, positive, negative)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # Scalar
        assert loss.item() >= 0
        assert loss.requires_grad
    
    def test_focal_weighting(self):
        """Test that focal loss applies proper weighting."""
        focal_loss = FocalTripletLoss(margin=1.0, alpha=1.0, gamma=2.0)
        
        # Create controlled case where we know the basic triplet loss
        anchor = torch.zeros(1, 2)
        positive = torch.tensor([[0.1, 0.0]])  # Close to anchor
        negative = torch.tensor([[0.2, 0.0]])  # Farther from anchor
        
        loss = focal_loss(anchor, positive, negative)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0


class TestMarginScheduling:
    """Test suite for margin scheduling functions."""
    
    def test_constant_schedule(self):
        """Test constant margin schedule."""
        schedule = create_margin_schedule('constant', max_epochs=100, initial_margin=1.0)
        
        assert schedule(0) == 1.0
        assert schedule(50) == 1.0
        assert schedule(100) == 1.0
    
    def test_linear_schedule(self):
        """Test linear margin schedule."""
        schedule = create_margin_schedule('linear', max_epochs=100, initial_margin=1.0, final_margin=0.5)
        
        assert schedule(0) == 1.0
        assert schedule(100) == 0.5
        assert 0.5 < schedule(50) < 1.0  # Should be between initial and final
    
    def test_cosine_schedule(self):
        """Test cosine margin schedule."""
        schedule = create_margin_schedule('cosine', max_epochs=100, initial_margin=1.0, final_margin=0.5)
        
        assert abs(schedule(0) - 1.0) < 1e-6
        assert abs(schedule(100) - 0.5) < 1e-6
        assert 0.5 <= schedule(50) <= 1.0
    
    def test_exponential_schedule(self):
        """Test exponential margin schedule."""
        schedule = create_margin_schedule('exponential', max_epochs=100, initial_margin=1.0, final_margin=0.5)
        
        assert abs(schedule(0) - 1.0) < 1e-6
        assert abs(schedule(100) - 0.5) < 1e-6
        # Exponential decay should be monotonic
        assert schedule(25) > schedule(75)
    
    def test_invalid_schedule(self):
        """Test invalid schedule type."""
        with pytest.raises(ValueError, match="Unknown schedule type"):
            create_margin_schedule('invalid')


class TestImprovedTrainer:
    """Test suite for ImprovedTrainer class."""
    
    @patch('train_improved.ImprovedCrowTripletDataset', MockImprovedCrowTripletDataset)
    @patch('train_improved.CrowResNetEmbedder', MockCrowResNetEmbedder)
    def test_trainer_initialization(self, basic_config, temp_training_dir):
        """Test ImprovedTrainer initialization."""
        config = basic_config.copy()
        config['crop_dir'] = temp_training_dir
        config['output_dir'] = os.path.join(temp_training_dir, 'output')
        
        # Import here to use the mocked modules
        from train_improved import ImprovedTrainer
        
        trainer = ImprovedTrainer(config)
        
        assert trainer.config == config
        assert trainer.device.type in ['cpu', 'cuda']
        assert isinstance(trainer.model, MockCrowResNetEmbedder)
        assert isinstance(trainer.criterion, ImprovedTripletLoss)
        assert trainer.start_epoch == 0
        assert trainer.best_separability == 0.0
        assert 'train_loss' in trainer.training_history
    
    @patch('train_improved.ImprovedCrowTripletDataset', MockImprovedCrowTripletDataset)
    @patch('train_improved.CrowResNetEmbedder', MockCrowResNetEmbedder)
    def test_load_datasets(self, basic_config, temp_training_dir):
        """Test dataset loading."""
        config = basic_config.copy()
        config['crop_dir'] = temp_training_dir
        config['output_dir'] = os.path.join(temp_training_dir, 'output')
        
        from train_improved import ImprovedTrainer
        
        trainer = ImprovedTrainer(config)
        
        assert hasattr(trainer, 'train_dataset')
        assert hasattr(trainer, 'val_dataset')
        assert hasattr(trainer, 'train_loader')
        assert hasattr(trainer, 'val_loader')
        assert len(trainer.train_dataset) > 0
        assert len(trainer.val_dataset) > 0
    
    @patch('train_improved.ImprovedCrowTripletDataset', MockImprovedCrowTripletDataset)
    @patch('train_improved.CrowResNetEmbedder', MockCrowResNetEmbedder)
    def test_checkpoint_saving_loading(self, basic_config, temp_training_dir):
        """Test checkpoint saving and loading."""
        config = basic_config.copy()
        config['crop_dir'] = temp_training_dir
        config['output_dir'] = os.path.join(temp_training_dir, 'output')
        
        from train_improved import ImprovedTrainer
        
        trainer = ImprovedTrainer(config)
        
        # Save checkpoint
        trainer.best_separability = 0.85
        trainer._save_checkpoint(5, is_best=True)
        
        # Check files exist
        checkpoint_dir = Path(trainer.output_dir) / 'checkpoints'
        assert (checkpoint_dir / 'latest_checkpoint.pth').exists()
        assert (checkpoint_dir / 'best_model.pth').exists()
        
        # Load checkpoint
        trainer2 = ImprovedTrainer(config)
        trainer2._load_checkpoint(checkpoint_dir / 'latest_checkpoint.pth')
        
        assert trainer2.start_epoch == 6  # epoch + 1
        assert trainer2.best_separability == 0.85
    
    @patch('train_improved.ImprovedCrowTripletDataset', MockImprovedCrowTripletDataset)
    @patch('train_improved.CrowResNetEmbedder', MockCrowResNetEmbedder)
    @patch('train_improved.tqdm')  # Mock progress bar
    def test_train_epoch(self, mock_tqdm, basic_config, temp_training_dir):
        """Test training for one epoch."""
        config = basic_config.copy()
        config['crop_dir'] = temp_training_dir
        config['output_dir'] = os.path.join(temp_training_dir, 'output')
        config['batch_size'] = 2  # Small batch for testing
        
        from train_improved import ImprovedTrainer
        
        # Mock tqdm to return our mock data loader
        mock_progress = Mock()
        mock_progress.__iter__ = lambda self: iter([
            # Mock batch data
            (
                (torch.randn(2, 3, 224, 224), torch.randn(2, 3, 224, 224), torch.randn(2, 3, 224, 224)),
                (torch.randn(2, 128, 64), torch.randn(2, 128, 64), torch.randn(2, 128, 64)),
                torch.randint(0, 3, (2,))
            )
        ])
        mock_progress.set_postfix = Mock()
        mock_tqdm.return_value = mock_progress
        
        trainer = ImprovedTrainer(config)
        loss = trainer.train_epoch(0)
        
        assert isinstance(loss, float)
        assert loss >= 0
        assert len(trainer.training_history['train_loss']) == 1
        assert len(trainer.training_history['learning_rates']) == 1
    
    @patch('train_improved.ImprovedCrowTripletDataset', MockImprovedCrowTripletDataset)
    @patch('train_improved.CrowResNetEmbedder', MockCrowResNetEmbedder)
    def test_simple_evaluation(self, basic_config, temp_training_dir):
        """Test simple evaluation method."""
        config = basic_config.copy()
        config['crop_dir'] = temp_training_dir
        config['output_dir'] = os.path.join(temp_training_dir, 'output')
        
        from train_improved import ImprovedTrainer
        
        trainer = ImprovedTrainer(config)
        separability = trainer.evaluate_simple(0)
        
        assert isinstance(separability, float)
        assert separability >= 0 or separability == 0.0  # Can be 0 for random embeddings
        assert len(trainer.training_history['eval_metrics']) <= 1  # May or may not add metrics
    
    @patch('train_improved.ImprovedCrowTripletDataset', MockImprovedCrowTripletDataset)
    @patch('train_improved.CrowResNetEmbedder', MockCrowResNetEmbedder)
    @patch('matplotlib.pyplot.savefig')  # Mock plot saving
    @patch('matplotlib.pyplot.close')
    def test_training_plots(self, mock_close, mock_savefig, basic_config, temp_training_dir):
        """Test training plot generation."""
        config = basic_config.copy()
        config['crop_dir'] = temp_training_dir
        config['output_dir'] = os.path.join(temp_training_dir, 'output')
        
        from train_improved import ImprovedTrainer
        
        trainer = ImprovedTrainer(config)
        
        # Add some mock training history
        trainer.training_history['train_loss'] = [1.0, 0.8, 0.6]
        trainer.training_history['learning_rates'] = [0.001, 0.0008, 0.0006]
        trainer.training_history['epochs'] = [1, 2, 3]
        trainer.training_history['eval_metrics'] = [
            {'epoch': 1, 'separability': 0.1, 'same_crow_sim': 0.7, 'diff_crow_sim': 0.6},
            {'epoch': 2, 'separability': 0.2, 'same_crow_sim': 0.8, 'diff_crow_sim': 0.6}
        ]
        
        trainer.save_training_plots()
        
        # Check that matplotlib functions were called
        mock_savefig.assert_called_once()
        mock_close.assert_called_once()
    
    @patch('train_improved.ImprovedCrowTripletDataset', MockImprovedCrowTripletDataset)
    @patch('train_improved.CrowResNetEmbedder', MockCrowResNetEmbedder)
    def test_training_loop_integration(self, basic_config, temp_training_dir):
        """Test the main training loop (mocked for speed)."""
        config = basic_config.copy()
        config['crop_dir'] = temp_training_dir
        config['output_dir'] = os.path.join(temp_training_dir, 'output')
        config['epochs'] = 2  # Very short training
        config['eval_every'] = 1
        config['plot_every'] = 1
        
        from train_improved import ImprovedTrainer
        
        trainer = ImprovedTrainer(config)
        
        # Mock the expensive parts
        with patch.object(trainer, 'train_epoch', return_value=0.5) as mock_train, \
             patch.object(trainer, 'evaluate_simple', return_value=0.3) as mock_eval, \
             patch.object(trainer, 'save_training_plots') as mock_plots:
            
            trainer.train()
            
            # Check that training methods were called
            assert mock_train.call_count == 2  # 2 epochs
            assert mock_eval.call_count == 2   # eval_every = 1
            assert mock_plots.call_count == 2  # plot_every = 1
            
            # Check that final model was saved
            final_model_path = Path(trainer.output_dir) / 'crow_resnet_triplet_improved.pth'
            assert final_model_path.exists()
    
    @patch('train_improved.ImprovedCrowTripletDataset', MockImprovedCrowTripletDataset)
    @patch('train_improved.CrowResNetEmbedder', MockCrowResNetEmbedder)
    def test_error_handling(self, basic_config, temp_training_dir):
        """Test error handling in training."""
        config = basic_config.copy()
        config['crop_dir'] = temp_training_dir
        config['output_dir'] = os.path.join(temp_training_dir, 'output')
        
        from train_improved import ImprovedTrainer
        
        trainer = ImprovedTrainer(config)
        
        # Test loading non-existent checkpoint
        trainer._load_checkpoint('nonexistent.pth')  # Should not crash
        
        # Test evaluation with corrupted data
        with patch.object(trainer, 'val_loader', []):  # Empty loader
            separability = trainer.evaluate_simple(0)
            assert separability == 0.0
    
    @patch('train_improved.ImprovedCrowTripletDataset', MockImprovedCrowTripletDataset)
    @patch('train_improved.CrowResNetEmbedder', MockCrowResNetEmbedder)
    def test_resume_from_checkpoint(self, basic_config, temp_training_dir):
        """Test resuming training from checkpoint."""
        config = basic_config.copy()
        config['crop_dir'] = temp_training_dir
        config['output_dir'] = os.path.join(temp_training_dir, 'output')
        
        from train_improved import ImprovedTrainer
        
        # Create initial trainer and save checkpoint
        trainer1 = ImprovedTrainer(config)
        trainer1.start_epoch = 5
        trainer1.best_separability = 0.7
        trainer1._save_checkpoint(5)
        
        # Create new trainer with resume config
        config['resume_from'] = str(Path(trainer1.output_dir) / 'checkpoints' / 'latest_checkpoint.pth')
        trainer2 = ImprovedTrainer(config)
        
        assert trainer2.start_epoch == 6  # Loaded epoch + 1
        assert trainer2.best_separability == 0.7


class TestSetupImprovedTraining:
    """Test suite for setup_improved_training.py functionality."""
    
    @patch('setup_improved_training.DatasetStats')
    def test_setup_training_config(self, mock_dataset_stats, temp_training_dir):
        """Test setup_improved_training configuration generation."""
        # Mock the dataset analysis
        mock_stats = {
            'total_crows': 5,
            'total_images': 50,
            'images_per_crow_stats': {'min': 5, 'max': 15, 'mean': 10.0, 'median': 10.0}
        }
        mock_recommendations = {
            'embedding_dim': 512,
            'epochs': 100,
            'batch_size': 16,
            'learning_rate': 0.0005
        }
        mock_dataset_stats.recommend_training_params.return_value = (mock_recommendations, mock_stats)
        
        from setup_improved_training import setup_training_config
        
        config_path = os.path.join(temp_training_dir, 'test_config.json')
        config = setup_training_config(temp_training_dir, config_path)
        
        assert os.path.exists(config_path)
        assert 'training_config' in config
        assert config['training_config']['embedding_dim'] == 512
        assert config['training_config']['epochs'] == 100
        
        # Verify the file was written correctly
        with open(config_path, 'r') as f:
            saved_config = json.load(f)
        assert saved_config == config


class TestIntegration:
    """Integration tests for the complete improved training system."""
    
    @patch('train_improved.ImprovedCrowTripletDataset', MockImprovedCrowTripletDataset)
    @patch('train_improved.CrowResNetEmbedder', MockCrowResNetEmbedder)
    def test_end_to_end_training(self, temp_training_dir):
        """Test complete end-to-end training workflow."""
        # Create configuration
        config = {
            'crop_dir': temp_training_dir,
            'embedding_dim': 128,  # Small for testing
            'epochs': 2,
            'batch_size': 4,
            'learning_rate': 0.01,
            'margin': 1.0,
            'mining_type': 'adaptive',
            'output_dir': os.path.join(temp_training_dir, 'training_output'),
            'eval_every': 1,
            'save_every': 1,
            'num_workers': 0,
            'early_stopping': False,
            'plot_every': 1
        }
        
        from train_improved import ImprovedTrainer
        
        # Run complete training
        trainer = ImprovedTrainer(config)
        
        # Mock the expensive parts but test the integration
        with patch.object(trainer, 'train_epoch', return_value=0.5) as mock_train, \
             patch.object(trainer, 'evaluate_simple', return_value=0.3) as mock_eval:
            
            trainer.train()
            
            # Verify the complete workflow executed
            assert mock_train.call_count == 2
            assert mock_eval.call_count == 2
            
            # Check all expected outputs
            output_dir = Path(config['output_dir'])
            assert (output_dir / 'crow_resnet_triplet_improved.pth').exists()
            assert (output_dir / 'training_history.json').exists()
            assert (output_dir / 'checkpoints' / 'latest_checkpoint.pth').exists()
    
    def test_all_mining_strategies_integration(self):
        """Test that all mining strategies work with the trainer."""
        mining_types = ['hard', 'semi_hard', 'adaptive', 'curriculum']
        
        for mining_type in mining_types:
            loss_fn = ImprovedTripletLoss(mining_type=mining_type)
            
            # Test with realistic data
            embeddings = torch.randn(16, 256)
            labels = torch.randint(0, 4, (16,))
            
            loss, stats = loss_fn(embeddings, labels)
            
            assert isinstance(loss, torch.Tensor)
            assert loss.requires_grad
            assert stats['mining_type'] == mining_type
            
            # Update epoch for curriculum learning
            if mining_type == 'curriculum':
                loss_fn.update_epoch(10)
                loss2, stats2 = loss_fn(embeddings, labels)
                assert stats2['difficulty'] > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v']) 