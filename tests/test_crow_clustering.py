import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from pathlib import Path
import json
import os
from crow_clustering import CrowClusterAnalyzer, process_video_for_clustering

@pytest.fixture
def sample_embeddings():
    """Create sample embeddings for testing."""
    # Create 3 distinct clusters with some noise
    n_samples = 30
    n_features = 512
    
    # Cluster 1: centered around (0.5, 0.5, ...)
    cluster1 = np.random.normal(0.5, 0.1, (10, n_features))
    # Cluster 2: centered around (0.8, 0.8, ...)
    cluster2 = np.random.normal(0.8, 0.1, (10, n_features))
    # Cluster 3: centered around (0.2, 0.2, ...)
    cluster3 = np.random.normal(0.2, 0.1, (8, n_features))
    # Noise points
    noise = np.random.uniform(0, 1, (2, n_features))
    
    # Combine all points
    embeddings = np.vstack([cluster1, cluster2, cluster3, noise])
    # Normalize each embedding
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    return embeddings

@pytest.fixture
def sample_frame_numbers():
    """Create sample frame numbers for temporal testing."""
    return list(range(30))  # 30 frames

@pytest.fixture
def sample_confidences():
    """Create sample confidence scores."""
    return [0.9] * 30  # High confidence for all samples

@pytest.fixture
def mock_db():
    """Mock database functions."""
    with patch('crow_clustering.get_crow_embeddings') as mock_get_embeddings, \
         patch('crow_clustering.get_all_crows') as mock_get_all_crows:
        
        # Mock get_crow_embeddings
        def mock_embeddings(crow_id):
            if crow_id == 999:  # Invalid crow_id
                return []
            return [{'embedding': np.random.rand(512).tolist()} for _ in range(5)]
        mock_get_embeddings.side_effect = mock_embeddings
        
        # Mock get_all_crows
        mock_get_all_crows.return_value = [
            {'id': i, 'embeddings': [np.random.rand(512).tolist() for _ in range(5)],
             'frame_number': i * 10, 'confidence': 0.9}
            for i in range(3)
        ]
        
        yield mock_get_embeddings, mock_get_all_crows

def test_crow_cluster_analyzer_init():
    """Test CrowClusterAnalyzer initialization."""
    analyzer = CrowClusterAnalyzer(
        eps_range=(0.2, 0.5),
        min_samples_range=(2, 5),
        similarity_threshold=0.7,
        temporal_weight=0.1
    )
    
    assert analyzer.eps_range == (0.2, 0.5)
    assert analyzer.min_samples_range == (2, 5)
    assert analyzer.similarity_threshold == 0.7
    assert analyzer.temporal_weight == 0.1
    assert analyzer.best_params is None
    assert analyzer.cluster_metrics == {}

def test_extract_embeddings(mock_db):
    """Test embedding extraction from database."""
    mock_get_embeddings, _ = mock_db
    analyzer = CrowClusterAnalyzer()
    
    # Test with valid crow_id
    embeddings = analyzer.extract_embeddings(1)
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape[1] == 512  # Embedding dimension
    assert len(embeddings) > 0
    
    # Test with invalid crow_id (should return empty array)
    embeddings = analyzer.extract_embeddings(999)
    assert isinstance(embeddings, np.ndarray)
    assert len(embeddings) == 0

def test_validate_clusters(sample_embeddings):
    """Test cluster validation."""
    analyzer = CrowClusterAnalyzer(similarity_threshold=0.7)
    
    # Create some cluster labels (3 clusters + noise)
    labels = np.array([0] * 10 + [1] * 10 + [2] * 8 + [-1] * 2)
    
    valid_clusters, metrics = analyzer.validate_clusters(labels, sample_embeddings)
    
    assert isinstance(valid_clusters, list)
    assert isinstance(metrics, dict)
    assert 'total_clusters' in metrics
    assert 'valid_clusters' in metrics
    assert 'invalid_clusters' in metrics
    assert 'cluster_details' in metrics
    
    # Check that metrics are reasonable
    assert metrics['total_clusters'] == 3
    assert metrics['valid_clusters'] + metrics['invalid_clusters'] == metrics['total_clusters']
    
    # Check cluster details
    for cluster_id in range(3):
        cluster_key = f'cluster_{cluster_id}'
        assert cluster_key in metrics['cluster_details']
        details = metrics['cluster_details'][cluster_key]
        assert 'size' in details
        assert 'mean_similarity' in details
        assert 'std_similarity' in details
        assert 'is_valid' in details

def test_find_optimal_params(sample_embeddings, sample_frame_numbers):
    """Test finding optimal clustering parameters."""
    analyzer = CrowClusterAnalyzer(
        eps_range=(0.2, 0.3),  # Narrow range for faster testing
        min_samples_range=(2, 3)
    )
    
    # Mock DBSCAN to ensure it returns valid clusters
    with patch('sklearn.cluster.DBSCAN') as mock_dbscan:
        mock_clustering = MagicMock()
        mock_clustering.labels_ = np.array([0] * 10 + [1] * 10 + [2] * 8 + [-1] * 2)
        mock_dbscan.return_value = mock_clustering
        
        # Mock validate_clusters to return valid metrics
        with patch.object(analyzer, 'validate_clusters') as mock_validate:
            mock_validate.return_value = (
                [0, 1, 2],  # valid_clusters
                {
                    'total_clusters': 3,
                    'valid_clusters': 3,
                    'invalid_clusters': 0,
                    'valid_cluster_ratio': 1.0,
                    'cluster_details': {
                        'cluster_0': {'size': 10, 'mean_similarity': 0.8, 'std_similarity': 0.1, 'is_valid': True},
                        'cluster_1': {'size': 10, 'mean_similarity': 0.8, 'std_similarity': 0.1, 'is_valid': True},
                        'cluster_2': {'size': 8, 'mean_similarity': 0.8, 'std_similarity': 0.1, 'is_valid': True}
                    }
                }
            )
            
            eps, min_samples = analyzer.find_optimal_params(sample_embeddings, sample_frame_numbers)
            
            assert isinstance(eps, float)
            assert isinstance(min_samples, int)
            assert analyzer.eps_range[0] <= eps <= analyzer.eps_range[1]
            assert analyzer.min_samples_range[0] <= min_samples <= analyzer.min_samples_range[1]
            
            # Check that metrics were saved
            assert 'parameter_search' in analyzer.cluster_metrics
            assert analyzer.best_params == (eps, min_samples)

def test_cluster_crows(sample_embeddings, sample_frame_numbers, sample_confidences):
    """Test crow clustering with various parameters."""
    analyzer = CrowClusterAnalyzer()
    
    # Mock DBSCAN to ensure it returns valid clusters
    with patch('sklearn.cluster.DBSCAN') as mock_dbscan:
        mock_clustering = MagicMock()
        mock_clustering.labels_ = np.array([0] * 10 + [1] * 10 + [2] * 8 + [-1] * 2)
        mock_dbscan.return_value = mock_clustering
        
        # Test with default parameters
        labels, metrics = analyzer.cluster_crows(
            sample_embeddings,
            frame_numbers=sample_frame_numbers,
            confidences=sample_confidences,
            eps=0.25,  # Provide explicit parameters to avoid parameter search
            min_samples=2
        )
        
        assert isinstance(labels, np.ndarray)
        assert len(labels) == len(sample_embeddings)
        assert isinstance(metrics, dict)
        assert 'n_clusters' in metrics
        assert 'validation_metrics' in metrics
        
        # Test with specific parameters
        labels, metrics = analyzer.cluster_crows(
            sample_embeddings,
            frame_numbers=sample_frame_numbers,
            confidences=sample_confidences,
            eps=0.25,
            min_samples=2
        )
        
        assert isinstance(labels, np.ndarray)
        assert len(labels) == len(sample_embeddings)
        
        # Test without temporal information
        labels, metrics = analyzer.cluster_crows(
            sample_embeddings,
            eps=0.25,
            min_samples=2
        )
        
        assert isinstance(labels, np.ndarray)
        assert len(labels) == len(sample_embeddings)

def test_compute_distance_matrix_with_temporal(sample_embeddings, sample_frame_numbers, sample_confidences):
    """Test distance matrix computation with temporal weighting."""
    analyzer = CrowClusterAnalyzer(temporal_weight=0.1)
    
    # Test with all parameters
    dist_matrix = analyzer.compute_distance_matrix_with_temporal(
        sample_embeddings,
        sample_frame_numbers,
        sample_confidences
    )
    
    assert isinstance(dist_matrix, np.ndarray)
    assert dist_matrix.shape == (len(sample_embeddings), len(sample_embeddings))
    assert np.all(dist_matrix >= 0)  # Distances should be non-negative
    assert np.all(dist_matrix <= 1)  # Normalized distances should be <= 1
    
    # Test without confidences
    dist_matrix = analyzer.compute_distance_matrix_with_temporal(
        sample_embeddings,
        sample_frame_numbers
    )
    
    assert isinstance(dist_matrix, np.ndarray)
    assert dist_matrix.shape == (len(sample_embeddings), len(sample_embeddings))

def test_process_video_for_clustering(mock_db, tmp_path):
    """Test video processing for clustering."""
    # Create a temporary video file
    video_path = tmp_path / "test_video.mp4"
    video_path.touch()
    
    # Mock DBSCAN to ensure it returns valid clusters
    with patch('sklearn.cluster.DBSCAN') as mock_dbscan:
        mock_clustering = MagicMock()
        mock_clustering.labels_ = np.array([0] * 10 + [1] * 10 + [2] * 8 + [-1] * 2)
        mock_dbscan.return_value = mock_clustering
        
        # Mock validate_clusters to return valid metrics
        with patch.object(CrowClusterAnalyzer, 'validate_clusters') as mock_validate:
            mock_validate.return_value = (
                [0, 1, 2],  # valid_clusters
                {
                    'total_clusters': 3,
                    'valid_clusters': 3,
                    'invalid_clusters': 0,
                    'valid_cluster_ratio': 1.0,
                    'cluster_details': {
                        'cluster_0': {'size': 10, 'mean_similarity': 0.8, 'std_similarity': 0.1, 'is_valid': 1},  # Use int instead of bool
                        'cluster_1': {'size': 10, 'mean_similarity': 0.8, 'std_similarity': 0.1, 'is_valid': 1},
                        'cluster_2': {'size': 8, 'mean_similarity': 0.8, 'std_similarity': 0.1, 'is_valid': 1}
                    }
                }
            )
            
            # Process the video
            results = process_video_for_clustering(
                str(video_path),
                str(tmp_path),
                eps=0.25,
                min_samples=2
            )
            
            assert isinstance(results, dict)
            assert 'parameters' in results
            assert 'metrics' in results
            assert 'clusters' in results
            
            # Check that output files were created
            assert (tmp_path / "clustering_results.json").exists()
            assert (tmp_path / "crow_clusters_visualization.png").exists()
            assert (tmp_path / "clustering_parameter_search.png").exists()
            
            # Check results file contents
            with open(tmp_path / "clustering_results.json") as f:
                saved_results = json.load(f)
            assert isinstance(saved_results, dict)
            assert 'parameters' in saved_results
            assert 'metrics' in saved_results
            assert 'clusters' in saved_results

def test_visualization_functions(sample_embeddings):
    """Test visualization functions."""
    analyzer = CrowClusterAnalyzer()
    
    # Create some cluster labels
    labels = np.array([0] * 10 + [1] * 10 + [2] * 8 + [-1] * 2)
    
    # Test cluster visualization
    with patch('matplotlib.pyplot.savefig') as mock_savefig:
        analyzer._visualize_clusters(sample_embeddings, labels)
        mock_savefig.assert_called_once()
    
    # Test parameter search visualization
    results = [
        {
            'eps': 0.25,
            'min_samples': 2,
            'n_clusters': 3,
            'n_noise': 2,
            'silhouette_score': 0.7,
            'calinski_harabasz_score': 100,
            'valid_cluster_ratio': 0.8,
            'combined_score': 0.75
        }
    ]
    
    with patch('matplotlib.pyplot.savefig') as mock_savefig:
        analyzer._plot_parameter_search(results)
        mock_savefig.assert_called_once()

def test_error_handling():
    """Test error handling in various functions."""
    analyzer = CrowClusterAnalyzer()
    
    # Test with empty embeddings
    with pytest.raises(ValueError, match="Expected 2D array"):
        analyzer.cluster_crows(np.array([]), eps=0.25, min_samples=2)
    
    # Test with invalid parameters
    with pytest.raises(ValueError, match="Invalid eps parameter"):
        analyzer.cluster_crows(np.random.rand(10, 512), eps=-1, min_samples=2)
    
    with pytest.raises(ValueError, match="Invalid min_samples parameter"):
        analyzer.cluster_crows(np.random.rand(10, 512), eps=0.25, min_samples=0)
    
    # Test with mismatched frame numbers
    with pytest.raises(ValueError, match="Number of frame numbers must match number of embeddings"):
        analyzer.cluster_crows(
            np.random.rand(10, 512),
            frame_numbers=[1, 2, 3],  # Mismatched length
            eps=0.25,
            min_samples=2
        )
    
    # Test with invalid confidence scores
    with pytest.raises(ValueError, match="Invalid confidence scores"):
        analyzer.cluster_crows(
            np.random.rand(10, 512),
            confidences=[-1] * 10,  # Invalid confidence
            eps=0.25,
            min_samples=2
        ) 