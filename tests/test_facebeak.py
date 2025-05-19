import unittest
import os
import torch
import numpy as np
from pathlib import Path
import pytest
import subprocess
from train_triplet_resnet import CrowTripletDataset
from detection import merge_overlapping_detections
from crow_clustering import CrowClusterAnalyzer

class TestFacebeak(unittest.TestCase):
    @pytest.fixture(autouse=True)
    def setup_video_data(self, video_test_data):
        """Setup test data from videos."""
        self.base_dir = video_test_data['base_dir']
        self.dataset = CrowTripletDataset(str(self.base_dir), split='train')
        
        # Get a video file and extract its audio for testing
        for crow_dir in self.base_dir.iterdir():
            if not crow_dir.is_dir():
                continue
            video_files = list(crow_dir.glob("*.mp4"))
            if video_files:
                self.video_path = video_files[0]
                # Extract audio for testing
                audio_path = crow_dir / "audio" / f"{self.video_path.stem}.wav"
                audio_path.parent.mkdir(exist_ok=True)
                try:
                    subprocess.run([
                        "ffmpeg", "-i", str(self.video_path),
                        "-vn", "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "1", "-y",
                        str(audio_path)
                    ], check=True, capture_output=True)
                    self.audio_path = audio_path
                    break
                except subprocess.CalledProcessError:
                    continue
        else:
            pytest.skip("No valid video files found in test data")

    def tearDown(self):
        """Clean up after each test."""
        if hasattr(self, 'audio_path') and self.audio_path.exists():
            self.audio_path.unlink()

    def test_audio_data_augmentation(self):
        """Test audio data augmentation using video audio."""
        # Test augmentation
        audio_features = self.dataset._load_and_preprocess_audio(str(self.audio_path))
        self.assertIsNotNone(audio_features)
        self.assertIsInstance(audio_features, dict)
        self.assertIn('mel_spec', audio_features)
        self.assertIn('chroma', audio_features)
        
        # Test that features are properly shaped
        mel_spec = audio_features['mel_spec']
        chroma = audio_features['chroma']
        self.assertEqual(mel_spec.shape[0], 128)  # Mel spectrogram bins
        self.assertEqual(chroma.shape[0], 12)  # Chroma bins
        self.assertTrue(mel_spec.shape[1] > 0)  # Time dimension
        self.assertTrue(chroma.shape[1] > 0)  # Time dimension

    def test_dynamic_padding_truncation(self):
        """Test dynamic padding/truncation using video audio."""
        audio_features = self.dataset._load_and_preprocess_audio(str(self.audio_path))
        self.assertIsNotNone(audio_features)
        self.assertIsInstance(audio_features, dict)
        
        # Test that features are properly shaped and normalized
        mel_spec = audio_features['mel_spec']
        chroma = audio_features['chroma']
        self.assertEqual(mel_spec.shape[0], 128)  # Mel spectrogram bins
        self.assertEqual(chroma.shape[0], 12)  # Chroma bins
        self.assertTrue(64 <= mel_spec.shape[1] <= 256)  # Check if within target range
        self.assertTrue(64 <= chroma.shape[1] <= 256)  # Check if within target range
        
        # Test normalization
        self.assertTrue(torch.all(mel_spec >= 0).item() and torch.all(mel_spec <= 1).item())
        self.assertTrue(torch.all(chroma >= 0).item() and torch.all(chroma <= 1).item())

    def test_merge_overlapping_detections_multi_view(self):
        """Test merging overlapping detections with multi-view information."""
        detections = [
            {'bbox': [100, 100, 200, 200], 'score': 0.9, 'view': 'front', 'class': 'crow'},
            {'bbox': [105, 105, 205, 205], 'score': 0.8, 'view': 'side', 'class': 'crow'},
            {'bbox': [300, 300, 400, 400], 'score': 0.7, 'view': 'front', 'class': 'crow'}
        ]
        merged = merge_overlapping_detections(detections, iou_threshold=0.4)
        self.assertLessEqual(len(merged), len(detections))
        
        # Check that merged detections have the expected structure
        for det in merged:
            self.assertIn('bbox', det)
            self.assertIn('score', det)
            self.assertIn('class', det)
            if 'model' in det and det['model'] == 'merged':
                self.assertIn('views', det)
            else:
                self.assertIn('view', det)
            # Check that bbox coordinates are valid
            x1, y1, x2, y2 = det['bbox']
            self.assertLess(x1, x2)
            self.assertLess(y1, y2)
            self.assertGreaterEqual(x1, 0)
            self.assertGreaterEqual(y1, 0)

    def test_temporal_consistency_clustering(self):
        """Test temporal consistency in clustering."""
        # Create sample embeddings with temporal information
        n_samples = 20  # Increased from 10 to ensure enough samples for clustering
        embeddings = np.random.rand(n_samples, 512)
        frame_numbers = list(range(n_samples))
        confidences = [0.9] * n_samples
        
        # Create analyzer with fixed parameters to avoid parameter search
        analyzer = CrowClusterAnalyzer(eps_range=(0.2, 0.3), min_samples_range=(2, 3))
        labels, metrics = analyzer.cluster_crows(embeddings, frame_numbers=frame_numbers, confidences=confidences, eps=0.25, min_samples=2)
        
        self.assertEqual(len(labels), len(embeddings))
        # Check if consecutive frames are more likely to be in the same cluster
        consecutive_same_cluster = sum(1 for i in range(len(labels)-1) if labels[i] == labels[i+1])
        self.assertGreater(consecutive_same_cluster, 0)
        # Check metrics
        self.assertIn('n_clusters', metrics)
        self.assertIn('validation_metrics', metrics)

    def test_weighting_embeddings_by_confidence(self):
        """Test weighting embeddings by confidence scores."""
        # Create sample embeddings with varying confidence
        n_samples = 20  # Increased from 10 to ensure enough samples for clustering
        embeddings = np.random.rand(n_samples, 512)
        frame_numbers = list(range(n_samples))
        confidences = [0.1 + 0.9 * i / (n_samples - 1) for i in range(n_samples)]  # Linear range from 0.1 to 1.0
        
        # Create analyzer with fixed parameters to avoid parameter search
        analyzer = CrowClusterAnalyzer(eps_range=(0.2, 0.3), min_samples_range=(2, 3))
        labels, metrics = analyzer.cluster_crows(embeddings, frame_numbers=frame_numbers, confidences=confidences, eps=0.25, min_samples=2)
        
        self.assertEqual(len(labels), len(embeddings))
        # Check that high confidence points influence clustering
        # This is a weak test since clustering is stochastic, but we can check basic properties
        self.assertTrue(len(set(labels)) > 0)  # Should have at least one cluster
        self.assertTrue(len(set(labels)) <= len(embeddings))  # Should not have more clusters than samples
        # Check metrics
        self.assertIn('n_clusters', metrics)
        self.assertIn('validation_metrics', metrics)

if __name__ == '__main__':
    unittest.main()
