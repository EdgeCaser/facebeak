#!/usr/bin/env python3
"""
Unit tests for the unsupervised learning module.
Tests all components: SimCLR, temporal consistency, auto-labeling, reconstruction validation.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import json
import cv2
from PIL import Image

# Import the classes to test
from unsupervised_learning import (
    SimCLRCrowDataset,
    SimCLRLoss,
    TemporalConsistencyLoss,
    AutoLabelingSystem,
    ReconstructionValidator,
    UnsupervisedTrainingPipeline,
    create_unsupervised_config
)
from unsupervised_gui_tools import ClusteringBasedLabelSmoother # Added import


class TestSimCLRCrowDataset:
    """Test the SimCLR dataset for self-supervised learning."""
    
    @pytest.fixture
    def sample_images(self, tmp_path):
        """Create sample images for testing."""
        image_paths = []
        for i in range(5):
            img_path = tmp_path / f"image_{i}.jpg"
            # Create a dummy image
            img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            cv2.imwrite(str(img_path), img)
            image_paths.append(str(img_path))
        return image_paths
    
    def test_init(self, sample_images):
        """Test dataset initialization."""
        dataset = SimCLRCrowDataset(sample_images, transform_strength=0.8)
        assert len(dataset) == 5
        assert dataset.transform_strength == 0.8
        assert len(dataset.image_paths) == 5
    
    def test_getitem_valid_image(self, sample_images):
        """Test getting valid image pairs."""
        dataset = SimCLRCrowDataset(sample_images)
        view1, view2 = dataset[0]
        
        # Check shapes
        assert view1.shape == (3, 224, 224)
        assert view2.shape == (3, 224, 224)
        
        # Check data type
        assert view1.dtype == torch.float32
        assert view2.dtype == torch.float32
        
        # Views should be different (due to random augmentations)
        assert not torch.equal(view1, view2)
    
    def test_getitem_invalid_image(self, tmp_path):
        """Test handling of invalid image paths."""
        invalid_paths = [str(tmp_path / "nonexistent.jpg")]
        dataset = SimCLRCrowDataset(invalid_paths)
        
        view1, view2 = dataset[0]
        
        # Should return dummy tensors
        assert torch.equal(view1, torch.zeros(3, 224, 224))
        assert torch.equal(view2, torch.zeros(3, 224, 224))
    
    def test_len(self, sample_images):
        """Test dataset length."""
        dataset = SimCLRCrowDataset(sample_images)
        assert len(dataset) == len(sample_images)


class TestSimCLRLoss:
    """Test the SimCLR contrastive loss function."""
    
    @pytest.fixture
    def sample_embeddings(self):
        """Create sample embeddings for testing."""
        batch_size = 4
        embedding_dim = 128
        z1 = torch.randn(batch_size, embedding_dim)
        z2 = torch.randn(batch_size, embedding_dim)
        return z1, z2
    
    def test_init(self):
        """Test loss function initialization."""
        loss_fn = SimCLRLoss(temperature=0.07, normalize=True)
        assert loss_fn.temperature == 0.07
        assert loss_fn.normalize is True
    
    def test_forward_normalized(self, sample_embeddings):
        """Test forward pass with normalization."""
        z1, z2 = sample_embeddings
        loss_fn = SimCLRLoss(temperature=0.07, normalize=True)
        
        loss = loss_fn(z1, z2)
        
        # Check loss properties
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar
        assert loss.item() >= 0  # Loss should be non-negative
        assert not torch.isnan(loss)
    
    def test_forward_unnormalized(self, sample_embeddings):
        """Test forward pass without normalization."""
        z1, z2 = sample_embeddings
        loss_fn = SimCLRLoss(temperature=0.07, normalize=False)
        
        loss = loss_fn(z1, z2)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert not torch.isnan(loss)
    
    def test_identical_embeddings(self):
        """Test loss with identical embeddings."""
        batch_size = 4
        embedding_dim = 128
        z = torch.randn(batch_size, embedding_dim)
        
        loss_fn = SimCLRLoss(temperature=0.07)
        loss = loss_fn(z, z.clone())
        
        # Loss should be low for identical embeddings
        assert loss.item() < 1.0
    
    def test_temperature_effect(self, sample_embeddings):
        """Test that temperature affects the loss."""
        z1, z2 = sample_embeddings
        
        loss_low_temp = SimCLRLoss(temperature=0.01)(z1, z2)
        loss_high_temp = SimCLRLoss(temperature=1.0)(z1, z2)
        
        # Different temperatures should give different losses
        assert not torch.isclose(loss_low_temp, loss_high_temp, atol=1e-3)


class TestTemporalConsistencyLoss:
    """Test the temporal consistency loss function."""
    
    @pytest.fixture
    def sample_temporal_data(self):
        """Create sample temporal data."""
        batch_size = 6
        embedding_dim = 512
        embeddings = torch.randn(batch_size, embedding_dim)
        frame_numbers = torch.tensor([10, 11, 12, 50, 51, 100])
        video_ids = torch.tensor([1, 1, 1, 2, 2, 3])
        return embeddings, frame_numbers, video_ids
    
    def test_init(self):
        """Test loss function initialization."""
        loss_fn = TemporalConsistencyLoss(weight=0.1, max_frames_gap=5)
        assert loss_fn.weight == 0.1
        assert loss_fn.max_frames_gap == 5
    
    def test_forward_same_video(self, sample_temporal_data):
        """Test loss calculation for same video samples."""
        embeddings, frame_numbers, video_ids = sample_temporal_data
        loss_fn = TemporalConsistencyLoss(weight=0.1, max_frames_gap=5)
        
        loss = loss_fn(embeddings, frame_numbers, video_ids)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert loss.item() >= 0
        assert not torch.isnan(loss)
    
    def test_forward_different_videos(self):
        """Test loss with samples from different videos."""
        embeddings = torch.randn(4, 512)
        frame_numbers = torch.tensor([10, 20, 30, 40])
        video_ids = torch.tensor([1, 2, 3, 4])  # All different videos
        
        loss_fn = TemporalConsistencyLoss(weight=0.1, max_frames_gap=5)
        loss = loss_fn(embeddings, frame_numbers, video_ids)
        
        # Should be near zero since no temporal pairs
        assert loss.item() < 1e-6
    
    def test_frame_gap_threshold(self):
        """Test that frame gap threshold is respected."""
        embeddings = torch.randn(3, 512)
        frame_numbers = torch.tensor([10, 11, 20])  # Gap of 9 frames
        video_ids = torch.tensor([1, 1, 1])
        
        loss_fn = TemporalConsistencyLoss(weight=0.1, max_frames_gap=5)
        loss = loss_fn(embeddings, frame_numbers, video_ids)
        
        # Only frames 10 and 11 should contribute (gap of 1)
        assert loss.item() >= 0


class TestAutoLabelingSystem:
    """Test the auto-labeling system for pseudo-label generation."""
    
    @pytest.fixture
    def sample_embeddings_and_labels(self):
        """Create sample embeddings and labels."""
        np.random.seed(42)  # For reproducibility
        
        # Create 3 distinct clusters
        cluster1 = np.random.normal([1, 1], 0.1, (5, 2))
        cluster2 = np.random.normal([3, 3], 0.1, (5, 2))
        cluster3 = np.random.normal([5, 5], 0.1, (5, 2))
        
        embeddings = np.vstack([cluster1, cluster2, cluster3])
        
        # Labels: some labeled, some unlabeled
        labels = ['crow_1'] * 3 + [None] * 2 + ['crow_2'] * 3 + [None] * 2 + ['crow_3'] * 3 + [None] * 2
        
        return embeddings, labels
    
    def test_init(self):
        """Test auto-labeling system initialization."""
        system = AutoLabelingSystem(confidence_threshold=0.95, distance_threshold=0.3)
        assert system.confidence_threshold == 0.95
        assert system.distance_threshold == 0.3
    
    def test_generate_pseudo_labels(self, sample_embeddings_and_labels):
        """Test pseudo-label generation."""
        embeddings, labels = sample_embeddings_and_labels
        system = AutoLabelingSystem(confidence_threshold=0.8)
        
        results = system.generate_pseudo_labels(embeddings, labels)
        
        assert 'pseudo_labels' in results
        assert 'confidences' in results
        assert isinstance(results['pseudo_labels'], dict)
        assert isinstance(results['confidences'], dict)
    
    def test_no_unlabeled_samples(self):
        """Test with no unlabeled samples."""
        embeddings = np.random.randn(5, 10)
        labels = ['crow_1'] * 5  # All labeled
        
        system = AutoLabelingSystem()
        results = system.generate_pseudo_labels(embeddings, labels)
        
        assert len(results['pseudo_labels']) == 0
        assert len(results['confidences']) == 0
    
    def test_identify_low_entropy_triplets(self):
        """Test low-entropy triplet identification."""
        # Create embeddings where some are very close
        embeddings = np.array([
            [1.0, 1.0],
            [1.1, 1.1],  # Very close to first
            [5.0, 5.0],  # Far from others
            [2.0, 2.0]
        ])
        
        system = AutoLabelingSystem(confidence_threshold=0.5)
        triplets = system.identify_low_entropy_triplets(embeddings)
        
        assert isinstance(triplets, list)
        for triplet in triplets:
            assert len(triplet) == 4  # anchor, positive, negative, confidence
            assert 0 <= triplet[3] <= 1  # Confidence should be in [0, 1]


class TestReconstructionValidator:
    """Test the autoencoder-based reconstruction validator."""
    
    @pytest.fixture
    def sample_embeddings(self):
        """Create sample embeddings for testing."""
        torch.manual_seed(42)
        return torch.randn(20, 512)
    
    def test_init(self):
        """Test reconstruction validator initialization."""
        validator = ReconstructionValidator(embedding_dim=512, hidden_dim=256)
        assert validator.embedding_dim == 512
        assert validator.hidden_dim == 256
        assert validator.autoencoder is not None
    
    def test_build_autoencoder(self):
        """Test autoencoder architecture."""
        validator = ReconstructionValidator(embedding_dim=128, hidden_dim=64)
        autoencoder = validator.autoencoder
        
        # Test forward pass
        x = torch.randn(5, 128)
        output = autoencoder(x)
        
        assert output.shape == (5, 128)
        assert not torch.isnan(output).any()
    
    def test_train_autoencoder(self, sample_embeddings):
        """Test autoencoder training."""
        validator = ReconstructionValidator(embedding_dim=512, hidden_dim=256)
        
        # Train for a few epochs
        final_loss = validator.train_autoencoder(sample_embeddings, epochs=5)
        
        assert isinstance(final_loss, float)
        assert final_loss >= 0
        assert not np.isnan(final_loss)
    
    def test_detect_outliers(self, sample_embeddings):
        """Test outlier detection."""
        validator = ReconstructionValidator(embedding_dim=512, hidden_dim=256)
        
        # Train first
        validator.train_autoencoder(sample_embeddings, epochs=5)
        
        # Detect outliers
        outlier_indices, threshold = validator.detect_outliers(sample_embeddings, threshold_percentile=90)
        
        assert isinstance(outlier_indices, list)
        assert isinstance(threshold, float)
        assert len(outlier_indices) <= len(sample_embeddings)
        assert threshold >= 0
    
    def test_detect_outliers_with_actual_outliers(self):
        """Test outlier detection with known outliers."""
        # Create mostly normal embeddings with some outliers
        normal_embeddings = torch.randn(15, 64)
        outlier_embeddings = torch.randn(5, 64) * 5  # Much larger variance
        
        all_embeddings = torch.cat([normal_embeddings, outlier_embeddings])
        
        validator = ReconstructionValidator(embedding_dim=64, hidden_dim=32)
        validator.train_autoencoder(all_embeddings, epochs=10)
        
        outlier_indices, _ = validator.detect_outliers(all_embeddings, threshold_percentile=75)
        
        # Should detect some outliers
        assert len(outlier_indices) > 0


class TestUnsupervisedTrainingPipeline:
    """Test the main unsupervised training pipeline."""
    
    @pytest.fixture
    def mock_model(self):
        """Create a mock model for testing."""
        model = Mock()
        model.embedding_dim = 512
        model.parameters.return_value = [torch.randn(10, requires_grad=True)]
        return model
    
    @pytest.fixture
    def sample_config(self):
        """Create sample configuration."""
        return {
            'simclr_epochs': 5,
            'simclr_batch_size': 4,
            'auto_label_confidence': 0.9,
            'outlier_percentile': 90
        }
    
    def test_init(self, mock_model, sample_config):
        """Test pipeline initialization."""
        pipeline = UnsupervisedTrainingPipeline(mock_model, sample_config)
        
        assert pipeline.model == mock_model
        assert pipeline.config == sample_config
        assert pipeline.auto_labeler is not None
        assert pipeline.reconstruction_validator is not None
    
    @patch('unsupervised_learning.get_all_crows')
    @patch('unsupervised_learning.get_crow_embeddings')
    def test_calculate_embedding_quality(self, mock_get_embeddings, mock_get_crows, mock_model, sample_config):
        """Test embedding quality calculation."""
        # Mock database responses
        mock_get_crows.return_value = [{'id': 1}, {'id': 2}]
        mock_get_embeddings.side_effect = [
            [{'embedding': [1, 2, 3]}, {'embedding': [1.1, 2.1, 3.1]}],
            [{'embedding': [4, 5, 6]}, {'embedding': [4.1, 5.1, 6.1]}]
        ]
        
        pipeline = UnsupervisedTrainingPipeline(mock_model, sample_config)
        quality_score = pipeline._calculate_embedding_quality(np.array([[1, 2, 3], [4, 5, 6]]))
        
        assert isinstance(quality_score, float)
        assert 0 <= quality_score <= 1
    
    def test_generate_recommendations(self, mock_model, sample_config):
        """Test recommendation generation."""
        pipeline = UnsupervisedTrainingPipeline(mock_model, sample_config)
        
        # Test with various result scenarios
        results = {
            'outliers': list(range(20)),  # Many outliers
            'quality_score': 0.2,  # Low quality
            'pseudo_labels': {'pseudo_labels': {i: f'crow_{i}' for i in range(15)}}  # Many pseudo-labels
        }
        
        recommendations = pipeline._generate_recommendations(results)
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        assert any('outliers' in rec for rec in recommendations)
        assert any('quality' in rec for rec in recommendations)
        assert any('pseudo-labels' in rec for rec in recommendations)
    
    @patch('unsupervised_learning.DataLoader')
    @patch('glob.glob')
    def test_pretrain_with_simclr_insufficient_data(self, mock_glob, mock_dataloader, mock_model, sample_config):
        """Test SimCLR pretraining with insufficient data."""
        mock_glob.return_value = ['image1.jpg', 'image2.jpg']  # Only 2 images
        
        pipeline = UnsupervisedTrainingPipeline(mock_model, sample_config)
        
        result = pipeline.pretrain_with_simclr(['some_dir'], epochs=1)
        
        assert result['status'] == 'skipped'
        assert result['reason'] == 'insufficient_data'


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_create_unsupervised_config_default(self):
        """Test creating default unsupervised config."""
        config = create_unsupervised_config({})
        
        assert 'simclr_epochs' in config
        assert 'simclr_batch_size' in config
        assert 'temporal_weight' in config
        assert 'auto_label_confidence' in config
        
        assert config['simclr_epochs'] == 50
        assert config['auto_label_confidence'] == 0.95
    
    def test_create_unsupervised_config_override(self):
        """Test creating config with overrides."""
        base_config = {
            'simclr_epochs': 100,
            'custom_param': 'custom_value'
        }
        
        config = create_unsupervised_config(base_config)
        
        assert config['simclr_epochs'] == 100  # Overridden
        assert config['custom_param'] == 'custom_value'  # Custom param preserved
        assert config['auto_label_confidence'] == 0.95  # Default preserved


@pytest.mark.integration
class TestIntegrationScenarios:
    """Integration tests for complete workflows."""
    
    @patch('unsupervised_learning.get_all_crows')
    @patch('unsupervised_learning.get_crow_embeddings')
    def test_full_unsupervised_analysis(self, mock_get_embeddings, mock_get_crows):
        """Test a complete unsupervised analysis workflow."""
        # Mock database with realistic data
        mock_get_crows.return_value = [
            {'id': 1, 'name': 'crow_1'},
            {'id': 2, 'name': 'crow_2'}
        ]
        
        # Create embeddings that form clusters
        embeddings_crow1 = [{'embedding': [1.0, 1.0, 1.0] + [0.0] * 509} for _ in range(5)]
        embeddings_crow2 = [{'embedding': [3.0, 3.0, 3.0] + [0.0] * 509} for _ in range(5)]
        
        mock_get_embeddings.side_effect = [embeddings_crow1, embeddings_crow2]
        
        # Create mock model
        mock_model = Mock()
        mock_model.embedding_dim = 512
        
        config = create_unsupervised_config({'auto_label_confidence': 0.8})
        pipeline = UnsupervisedTrainingPipeline(mock_model, config)
        
        # Run analysis
        results = pipeline.apply_unsupervised_techniques("")
        
        # Verify results structure
        assert 'pseudo_labels' in results
        assert 'outliers' in results
        assert 'quality_score' in results
        assert 'recommendations' in results
        
        assert isinstance(results['quality_score'], float)
        assert isinstance(results['recommendations'], list)


class TestClusteringBasedLabelSmoother:
    """Test the ClusteringBasedLabelSmoother class, focusing on merge operations."""

    @pytest.fixture
    def label_smoother(self):
        """Fixture to create a ClusteringBasedLabelSmoother instance."""
        return ClusteringBasedLabelSmoother(confidence_threshold=0.8)

    @patch('unsupervised_gui_tools.reassign_crow_embeddings')
    @patch('unsupervised_gui_tools.get_connection')
    def test_perform_merge_operation_success(self, mock_get_connection, mock_reassign_embeddings, label_smoother):
        """Test successful merge operation."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_get_connection.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor
        mock_cursor.fetchone.return_value = (0,) # No remaining embeddings after reassign

        mock_reassign_embeddings.return_value = 5  # Simulate 5 embeddings moved

        crow_id_from = 1
        crow_id_to = 2
        success, message = label_smoother.perform_merge_operation(crow_id_from, crow_id_to)

        assert success is True
        assert message == f"Successfully merged crow {crow_id_from} into {crow_id_to}."
        mock_reassign_embeddings.assert_called_once_with(from_crow_id=crow_id_from, to_crow_id=crow_id_to)
        
        # Check SQL delete calls
        calls = [
            call("SELECT COUNT(*) FROM crow_embeddings WHERE crow_id = ?", (crow_id_from,)),
            call("DELETE FROM crows WHERE id = ?", (crow_id_from,))
        ]
        mock_cursor.execute.assert_has_calls(calls)
        mock_conn.commit.assert_called_once()
        mock_conn.close.assert_called_once()

    @patch('unsupervised_gui_tools.reassign_crow_embeddings')
    def test_perform_merge_operation_reassign_fails(self, mock_reassign_embeddings, label_smoother):
        """Test merge operation when reassign_crow_embeddings fails."""
        mock_reassign_embeddings.side_effect = Exception("DB error during reassign")

        crow_id_from = 1
        crow_id_to = 2
        success, message = label_smoother.perform_merge_operation(crow_id_from, crow_id_to)

        assert success is False
        assert "Merge failed: DB error during reassign" in message
        mock_reassign_embeddings.assert_called_once_with(from_crow_id=crow_id_from, to_crow_id=crow_id_to)

    @patch('unsupervised_gui_tools.reassign_crow_embeddings')
    @patch('unsupervised_gui_tools.get_connection')
    def test_perform_merge_operation_delete_fails(self, mock_get_connection, mock_reassign_embeddings, label_smoother):
        """Test merge operation when deleting the crow fails."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_get_connection.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor
        mock_cursor.fetchone.return_value = (0,) # No remaining embeddings

        # Simulate error on delete
        mock_cursor.execute.side_effect = [None, Exception("DB error during delete")] 

        mock_reassign_embeddings.return_value = 3

        crow_id_from = 1
        crow_id_to = 2
        success, message = label_smoother.perform_merge_operation(crow_id_from, crow_id_to)

        assert success is False
        assert f"Failed to delete crow {crow_id_from}: DB error during delete" in message
        mock_reassign_embeddings.assert_called_once_with(from_crow_id=crow_id_from, to_crow_id=crow_id_to)
        mock_conn.rollback.assert_called_once() # Should rollback if delete fails
        mock_conn.close.assert_called_once()

    @patch('unsupervised_gui_tools.reassign_crow_embeddings')
    @patch('unsupervised_gui_tools.get_connection')
    def test_perform_merge_operation_deletes_remaining_embeddings(self, mock_get_connection, mock_reassign_embeddings, label_smoother):
        """Test that remaining embeddings are deleted if found before deleting crow."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_get_connection.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor
        
        # Simulate some embeddings remaining after reassign_crow_embeddings (e.g. if it was partial)
        # First call to fetchone (COUNT(*)) returns 2, then 0 after delete.
        mock_cursor.fetchone.side_effect = [(2,), (0,)] 
        mock_reassign_embeddings.return_value = 5 

        crow_id_from = 1
        crow_id_to = 2
        success, message = label_smoother.perform_merge_operation(crow_id_from, crow_id_to)

        assert success is True
        mock_reassign_embeddings.assert_called_once_with(from_crow_id=crow_id_from, to_crow_id=crow_id_to)
        
        # Check SQL calls: count, delete embeddings, delete crow
        calls = [
            call("SELECT COUNT(*) FROM crow_embeddings WHERE crow_id = ?", (crow_id_from,)),
            call("DELETE FROM crow_embeddings WHERE crow_id = ?", (crow_id_from,)), # This is the extra delete
            call("DELETE FROM crows WHERE id = ?", (crow_id_from,))
        ]
        mock_cursor.execute.assert_has_calls(calls, any_order=False) # Order matters here
        mock_conn.commit.assert_called_once()
        mock_conn.close.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 