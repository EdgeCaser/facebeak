#!/usr/bin/env python3
"""
Unsupervised Learning Module for Facebeak
Implements advanced unsupervised techniques to improve crow embeddings and reduce labeling effort.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
import random
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path
from sklearn.cluster import DBSCAN
from collections import defaultdict
import json

from db import get_crow_embeddings, get_all_crows
from models import CrowResNetEmbedder

logger = logging.getLogger(__name__)


class SimCLRCrowDataset(Dataset):
    """Dataset for self-supervised pretraining using SimCLR on crow crops."""
    
    def __init__(self, image_paths: List[str], transform_strength: float = 1.0):
        self.image_paths = image_paths
        self.transform_strength = transform_strength
        
        # Strong augmentations for contrastive learning
        self.augment = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(512, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(
                brightness=0.4 * transform_strength,
                contrast=0.4 * transform_strength,
                saturation=0.4 * transform_strength,
                hue=0.1 * transform_strength
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = cv2.imread(str(img_path))
        if image is None:
            # Return a dummy tensor if image loading fails
            return torch.zeros(3, 512, 512), torch.zeros(3, 512, 512)
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Create two different augmented views
        view1 = self.augment(image)
        view2 = self.augment(image)
        
        return view1, view2


class SimCLRLoss(nn.Module):
    """SimCLR contrastive loss for self-supervised learning."""
    
    def __init__(self, temperature: float = 0.07, normalize: bool = True):
        super().__init__()
        self.temperature = temperature
        self.normalize = normalize
        
    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        Compute SimCLR loss between two augmented views.
        
        Args:
            z1, z2: (batch_size, embedding_dim) tensors
        """
        batch_size = z1.shape[0]
        
        if self.normalize:
            z1 = F.normalize(z1, dim=1)
            z2 = F.normalize(z2, dim=1)
        
        # Concatenate representations
        z = torch.cat([z1, z2], dim=0)  # (2*batch_size, embedding_dim)
        
        # Compute similarity matrix
        sim_matrix = torch.mm(z, z.t()) / self.temperature
        
        # Create labels for contrastive learning
        labels = torch.cat([torch.arange(batch_size), torch.arange(batch_size)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float().to(z.device)
        
        # Remove diagonal (self-similarity)
        labels = labels[~torch.eye(labels.shape[0], dtype=bool)].view(labels.shape[0], -1)
        sim_matrix = sim_matrix[~torch.eye(sim_matrix.shape[0], dtype=bool)].view(sim_matrix.shape[0], -1)
        
        # Compute loss
        positives = sim_matrix[labels.bool()].view(labels.shape[0], -1)
        negatives = sim_matrix[~labels.bool()].view(sim_matrix.shape[0], -1)
        
        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(z.device)
        
        loss = F.cross_entropy(logits, labels)
        return loss


class TemporalConsistencyLoss(nn.Module):
    """Temporal consistency loss to enforce smooth embeddings across time."""
    
    def __init__(self, weight: float = 0.1, max_frames_gap: int = 5):
        super().__init__()
        self.weight = weight
        self.max_frames_gap = max_frames_gap
        
    def forward(self, embeddings: torch.Tensor, frame_numbers: torch.Tensor, 
                video_ids: torch.Tensor) -> torch.Tensor:
        """
        Compute temporal consistency loss.
        
        Args:
            embeddings: (batch_size, embedding_dim)
            frame_numbers: (batch_size,) frame numbers
            video_ids: (batch_size,) video identifiers
        """
        loss = torch.tensor(0.0, device=embeddings.device)
        count = 0
        
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                # Only consider samples from same video
                if video_ids[i] != video_ids[j]:
                    continue
                
                frame_diff = abs(frame_numbers[i] - frame_numbers[j])
                
                # Apply temporal weighting
                if frame_diff <= self.max_frames_gap:
                    weight = 1.0 / (1.0 + frame_diff)
                    temporal_loss = F.mse_loss(embeddings[i], embeddings[j])
                    loss += weight * temporal_loss
                    count += 1
        
        return self.weight * loss / max(count, 1)


class AutoLabelingSystem:
    """System for auto-labeling low-entropy triplets and pseudo-label generation."""
    
    def __init__(self, confidence_threshold: float = 0.95, 
                 distance_threshold: float = 0.3):
        self.confidence_threshold = confidence_threshold
        self.distance_threshold = distance_threshold
        
    def generate_pseudo_labels(self, embeddings: np.ndarray, 
                             existing_labels: List[str]) -> Dict:
        """
        Generate pseudo-labels for unlabeled data based on clustering.
        
        Args:
            embeddings: (N, embedding_dim) array
            existing_labels: List of current labels (None for unlabeled)
            
        Returns:
            Dictionary with pseudo-label assignments and confidence scores
        """
        # Identify unlabeled samples
        unlabeled_indices = [i for i, label in enumerate(existing_labels) 
                           if label is None or label == 'unlabeled']
        
        if not unlabeled_indices:
            return {'pseudo_labels': {}, 'confidences': {}}
        
        # Run clustering on all embeddings
        clustering = DBSCAN(eps=0.5, min_samples=3)
        cluster_labels = clustering.fit_predict(embeddings)
        
        pseudo_labels = {}
        confidences = {}
        
        for idx in unlabeled_indices:
            cluster_id = cluster_labels[idx]
            
            if cluster_id == -1:  # Noise point
                continue
            
            # Find other points in the same cluster with existing labels
            cluster_members = np.where(cluster_labels == cluster_id)[0]
            labeled_members = [i for i in cluster_members 
                             if i not in unlabeled_indices and existing_labels[i]]
            
            if not labeled_members:
                continue
            
            # Check label consistency within cluster
            member_labels = [existing_labels[i] for i in labeled_members]
            unique_labels = list(set(member_labels))
            
            if len(unique_labels) == 1:  # Consistent labeling
                # Calculate confidence based on distance to cluster center
                cluster_embeddings = embeddings[cluster_members]
                center = np.mean(cluster_embeddings, axis=0)
                distance = np.linalg.norm(embeddings[idx] - center)
                
                confidence = 1.0 / (1.0 + distance)
                
                if confidence > self.confidence_threshold:
                    pseudo_labels[idx] = unique_labels[0]
                    confidences[idx] = confidence
        
        return {'pseudo_labels': pseudo_labels, 'confidences': confidences}
    
    def identify_low_entropy_triplets(self, embeddings: np.ndarray) -> List[Tuple]:
        """
        Identify triplets with low entropy (high confidence) for auto-labeling.
        
        Returns:
            List of (anchor_idx, positive_idx, negative_idx, confidence) tuples
        """
        n_samples = len(embeddings)
        low_entropy_triplets = []
        
        for i in range(n_samples):
            # Find nearest and farthest neighbors
            distances = np.linalg.norm(embeddings - embeddings[i], axis=1)
            distances[i] = np.inf  # Exclude self
            
            nearest_idx = np.argmin(distances)
            farthest_idx = np.argmax(distances[distances != np.inf])
            
            # Calculate entropy/confidence
            nearest_dist = distances[nearest_idx]
            second_nearest_dist = np.partition(distances, 1)[1]
            
            # High confidence if nearest neighbor is much closer than second nearest
            confidence = (second_nearest_dist - nearest_dist) / second_nearest_dist
            
            if confidence > self.confidence_threshold:
                low_entropy_triplets.append((i, nearest_idx, farthest_idx, confidence))
        
        return low_entropy_triplets


class ReconstructionValidator:
    """Auto-encoder based reconstruction validator for detecting outliers."""
    
    def __init__(self, embedding_dim: int = 512, hidden_dim: int = 256):
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.autoencoder = self._build_autoencoder()
        
    def _build_autoencoder(self) -> nn.Module:
        """Build a simple autoencoder for embedding reconstruction."""
        return nn.Sequential(
            # Encoder
            nn.Linear(self.embedding_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            # Decoder
            nn.Linear(self.hidden_dim // 2, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.embedding_dim)
        )
    
    def train_autoencoder(self, embeddings: torch.Tensor, epochs: int = 100) -> float:
        """
        Train autoencoder on embeddings.
        
        Returns:
            Final reconstruction loss
        """
        optimizer = torch.optim.Adam(self.autoencoder.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        self.autoencoder.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            reconstructed = self.autoencoder(embeddings)
            loss = criterion(reconstructed, embeddings)
            loss.backward()
            optimizer.step()
            
            if epoch % 20 == 0:
                logger.info(f"Autoencoder epoch {epoch}, loss: {loss.item():.4f}")
        
        return loss.item()
    
    def detect_outliers(self, embeddings: torch.Tensor, 
                       threshold_percentile: float = 95) -> Tuple[List[int], float]:
        """
        Detect outliers based on reconstruction error.
        
        Returns:
            (outlier_indices, threshold_used)
        """
        self.autoencoder.eval()
        with torch.no_grad():
            reconstructed = self.autoencoder(embeddings)
            reconstruction_errors = F.mse_loss(reconstructed, embeddings, reduction='none')
            reconstruction_errors = reconstruction_errors.mean(dim=1).cpu().numpy()
        
        threshold = np.percentile(reconstruction_errors, threshold_percentile)
        outlier_indices = np.where(reconstruction_errors > threshold)[0].tolist()
        
        return outlier_indices, threshold


class UnsupervisedTrainingPipeline:
    """Main pipeline orchestrating all unsupervised learning techniques."""
    
    def __init__(self, model: CrowResNetEmbedder, config: Dict):
        self.model = model
        self.config = config
        self.auto_labeler = AutoLabelingSystem(
            confidence_threshold=config.get('auto_label_confidence', 0.95)
        )
        self.reconstruction_validator = ReconstructionValidator()
        
    def pretrain_with_simclr(self, unlabeled_image_paths: List[str], 
                           epochs: int = 50) -> Dict:
        """
        Self-supervised pretraining using SimCLR.
        
        Returns:
            Training statistics
        """
        logger.info(f"Starting SimCLR pretraining on {len(unlabeled_image_paths)} images")
        
        dataset = SimCLRCrowDataset(unlabeled_image_paths)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
        
        # Add projection head for contrastive learning
        projection_head = nn.Sequential(
            nn.Linear(self.model.embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )
        
        optimizer = torch.optim.Adam(
            list(self.model.parameters()) + list(projection_head.parameters()),
            lr=0.001
        )
        
        criterion = SimCLRLoss(temperature=0.07)
        
        stats = {'losses': [], 'epochs': epochs}
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            self.model.train()
            projection_head.train()
            
            for batch_idx, (view1, view2) in enumerate(dataloader):
                optimizer.zero_grad()
                
                # Get embeddings for both views
                z1 = self.model(view1)
                z2 = self.model(view2)
                
                # Project to contrastive space
                z1_proj = projection_head(z1)
                z2_proj = projection_head(z2)
                
                # Compute contrastive loss
                loss = criterion(z1_proj, z2_proj)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                
                if batch_idx % 100 == 0:
                    logger.info(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
            
            avg_loss = epoch_loss / len(dataloader)
            stats['losses'].append(avg_loss)
            logger.info(f"Epoch {epoch} completed, average loss: {avg_loss:.4f}")
        
        # Remove projection head for downstream tasks
        del projection_head
        
        return stats
    
    def apply_unsupervised_techniques(self, video_path: str) -> Dict:
        """
        Apply all unsupervised learning techniques to improve embeddings.
        
        Returns:
            Results dictionary with all technique outputs
        """
        results = {
            'pseudo_labels': {},
            'outliers': [],
            'quality_score': 0.0,
            'recommendations': []
        }
        
        # Get existing embeddings and labels
        crows = get_all_crows()
        all_embeddings = []
        all_labels = []
        
        for crow in crows:
            embeddings = get_crow_embeddings(crow['id'])
            for emb in embeddings:
                all_embeddings.append(emb['embedding'])
                all_labels.append(str(crow['id']))
        
        if not all_embeddings:
            logger.warning("No embeddings found for unsupervised analysis")
            return results
        
        embeddings_array = np.array(all_embeddings)
        try:
            del all_embeddings # Free memory from the list of embeddings as it's now in embeddings_array
            logger.debug("Deleted all_embeddings list after converting to numpy array.")
        except NameError:
            pass # Should exist at this point if code logic is correct
        
        # 1. Generate pseudo-labels
        pseudo_results = self.auto_labeler.generate_pseudo_labels(
            embeddings_array, all_labels
        )
        results['pseudo_labels'] = pseudo_results
        
        # 2. Detect outliers with autoencoder
        embeddings_tensor = torch.tensor(embeddings_array, dtype=torch.float32)
        self.reconstruction_validator.train_autoencoder(embeddings_tensor)
        outlier_indices, threshold = self.reconstruction_validator.detect_outliers(
            embeddings_tensor
        )
        results['outliers'] = outlier_indices
        results['outlier_threshold'] = threshold
        
        # 3. Calculate overall quality score
        results['quality_score'] = self._calculate_embedding_quality(embeddings_array)
        
        # 4. Generate recommendations
        results['recommendations'] = self._generate_recommendations(results)

        # Clean up large data structures
        try:
            # all_embeddings should have been deleted earlier
            del embeddings_array
            del embeddings_tensor # Defined in this scope
            logger.debug("Cleaned up large embedding structures (array and tensor) in apply_unsupervised_techniques")
        except NameError:
            # embeddings_tensor might not always be defined if there are no embeddings
            logger.debug("Some embedding structures (array or tensor) were not defined, skipping cleanup for them.")
            pass # In case any of them were not defined (e.g. no embeddings found)

        return results
    
    def _calculate_embedding_quality(self, embeddings: np.ndarray) -> float:
        """Calculate overall embedding space quality score."""
        # Use silhouette score as a proxy for embedding quality
        try:
            from sklearn.metrics import silhouette_score
            clustering = DBSCAN(eps=0.5, min_samples=3)
            labels = clustering.fit_predict(embeddings)
            
            if len(set(labels)) > 1:
                score = silhouette_score(embeddings, labels)
                return max(0.0, score)  # Normalize to [0, 1]
            else:
                return 0.0
        except Exception as e:
            logger.warning(f"Could not calculate quality score: {e}")
            return 0.0
    
    def _generate_recommendations(self, results: Dict) -> List[str]:
        """Generate actionable recommendations based on analysis results."""
        recommendations = []
        
        if len(results['outliers']) > len(results.get('pseudo_labels', {})) * 0.1:
            recommendations.append(
                "High number of outliers detected. Consider reviewing labeling quality."
            )
        
        if results['quality_score'] < 0.3:
            recommendations.append(
                "Low embedding quality. Consider more diverse training data or "
                "self-supervised pretraining."
            )
        
        if len(results.get('pseudo_labels', {}).get('pseudo_labels', {})) > 10:
            recommendations.append(
                f"Found {len(results['pseudo_labels']['pseudo_labels'])} confident "
                "pseudo-labels ready for review and training data expansion."
            )
        
        return recommendations


def create_unsupervised_config(base_config: Dict) -> Dict:
    """Create configuration for unsupervised learning pipeline."""
    default_config = {
        'simclr_epochs': 50,
        'simclr_batch_size': 32,
        'simclr_temperature': 0.07,
        'temporal_weight': 0.1,
        'auto_label_confidence': 0.95,
        'distance_threshold': 0.3,
        'outlier_percentile': 95,
        'max_frames_gap': 5
    }
    
    return {**default_config, **base_config} 