#!/usr/bin/env python3
"""
Enhanced Training with Unsupervised Learning
Integrates self-supervised pretraining, temporal consistency, and auto-labeling
with the existing triplet loss training pipeline.
"""

import torch
import torch.nn as nn
import numpy as np
import argparse
import logging
import json
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime
import glob

from models import CrowResNetEmbedder
from improved_triplet_loss import ImprovedTripletLoss
from unsupervised_learning import (
    UnsupervisedTrainingPipeline, 
    TemporalConsistencyLoss,
    create_unsupervised_config
)
from train_triplet_resnet import CrowTripletDataset
from db import get_all_crows, get_crow_embeddings
from crow_clustering import CrowClusterAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('unsupervised_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class EnhancedTripletDataset(CrowTripletDataset):
    """Enhanced triplet dataset with temporal information for consistency loss."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.temporal_data = self._collect_temporal_data()
    
    def _collect_temporal_data(self) -> Dict:
        """Collect frame numbers and video IDs for temporal consistency."""
        temporal_info = {}
        
        for crow_id in self.crow_data:
            temporal_info[crow_id] = []
            embeddings = get_crow_embeddings(crow_id)
            
            for emb in embeddings:
                temporal_info[crow_id].append({
                    'frame_number': emb.get('frame_number', 0),
                    'video_path': emb.get('video_path', ''),
                    'embedding_path': emb.get('image_path', '')
                })
        
        return temporal_info
    
    def get_temporal_batch_data(self, batch_indices: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get frame numbers and video IDs for a batch."""
        frame_numbers = []
        video_ids = []
        
        for idx in batch_indices:
            # Extract crow_id and image_idx from the dataset structure
            crow_id = self.triplet_indices[idx][0]  # Anchor crow ID
            
            # Find temporal data for this sample
            temporal_data = self.temporal_data.get(crow_id, [])
            if temporal_data:
                # Use first available temporal data (could be improved with exact matching)
                frame_numbers.append(temporal_data[0]['frame_number'])
                video_ids.append(hash(temporal_data[0]['video_path']) % 10000)  # Simple video ID
            else:
                frame_numbers.append(0)
                video_ids.append(0)
        
        return torch.tensor(frame_numbers), torch.tensor(video_ids)


class UnsupervisedEnhancedTrainer:
    """Training pipeline with integrated unsupervised learning techniques."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model
        self.model = CrowResNetEmbedder(
            embedding_dim=config.get('embedding_dim', 512),
            dropout_rate=config.get('dropout_rate', 0.1)
        ).to(self.device)
        
        # Initialize loss functions
        self.triplet_loss = ImprovedTripletLoss(
            margin=config.get('margin', 1.0),
            mining_type=config.get('mining_type', 'adaptive')
        )
        
        self.temporal_loss = TemporalConsistencyLoss(
            weight=config.get('temporal_weight', 0.1),
            max_frames_gap=config.get('max_frames_gap', 5)
        )
        
        # Initialize unsupervised pipeline
        self.unsupervised_pipeline = UnsupervisedTrainingPipeline(
            self.model, 
            create_unsupervised_config(config)
        )
        
        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.get('learning_rate', 0.001),
            weight_decay=config.get('weight_decay', 0.01)
        )
        
        # Initialize scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        self.training_stats = {
            'epochs': [],
            'triplet_losses': [],
            'temporal_losses': [],
            'total_losses': [],
            'clustering_scores': [],
            'pseudo_labels_generated': []
        }
    
    def run_self_supervised_pretraining(self, crop_directories: List[str]) -> Dict:
        """
        Run self-supervised pretraining on unlabeled crop images.
        
        Args:
            crop_directories: List of directories containing crop images
        
        Returns:
            Pretraining statistics
        """
        logger.info("🧠 Starting self-supervised pretraining phase...")
        
        # Collect all unlabeled images
        unlabeled_paths = []
        for crop_dir in crop_directories:
            patterns = ['*.jpg', '*.jpeg', '*.png']
            for pattern in patterns:
                unlabeled_paths.extend(glob.glob(f"{crop_dir}/**/{pattern}", recursive=True))
        
        logger.info(f"Found {len(unlabeled_paths)} unlabeled images for pretraining")
        
        if len(unlabeled_paths) < 100:
            logger.warning("Not enough unlabeled images for effective pretraining")
            return {'status': 'skipped', 'reason': 'insufficient_data'}
        
        # Run SimCLR pretraining
        pretraining_stats = self.unsupervised_pipeline.pretrain_with_simclr(
            unlabeled_paths,
            epochs=self.config.get('simclr_epochs', 50)
        )
        
        logger.info("✅ Self-supervised pretraining completed")
        return pretraining_stats
    
    def train_epoch(self, dataloader, epoch: int) -> Dict:
        """Train for one epoch with unsupervised enhancements."""
        self.model.train()
        
        epoch_triplet_loss = 0.0
        epoch_temporal_loss = 0.0
        epoch_total_loss = 0.0
        
        for batch_idx, (anchors, positives, negatives) in enumerate(dataloader):
            anchors = anchors.to(self.device)
            positives = positives.to(self.device)
            negatives = negatives.to(self.device)
            
            # Get embeddings
            anchor_embeddings = self.model(anchors)
            positive_embeddings = self.model(positives)
            negative_embeddings = self.model(negatives)
            
            # Combine all embeddings for triplet loss
            all_embeddings = torch.cat([anchor_embeddings, positive_embeddings, negative_embeddings], dim=0)
            all_labels = torch.cat([
                torch.arange(len(anchors)),
                torch.arange(len(anchors)),  # Same labels for positives
                torch.arange(len(anchors), len(anchors) * 2)  # Different labels for negatives
            ]).to(self.device)
            
            # Compute triplet loss
            triplet_loss, triplet_stats = self.triplet_loss(all_embeddings, all_labels)
            
            # Compute temporal consistency loss
            if hasattr(dataloader.dataset, 'get_temporal_batch_data'):
                batch_indices = list(range(batch_idx * dataloader.batch_size, 
                                         min((batch_idx + 1) * dataloader.batch_size, 
                                             len(dataloader.dataset))))
                frame_numbers, video_ids = dataloader.dataset.get_temporal_batch_data(batch_indices)
                frame_numbers = frame_numbers.to(self.device)
                video_ids = video_ids.to(self.device)
                
                temporal_loss = self.temporal_loss(anchor_embeddings, frame_numbers, video_ids)
            else:
                temporal_loss = torch.tensor(0.0, device=self.device)
            
            # Combine losses
            total_loss = triplet_loss + temporal_loss
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Update epoch statistics
            epoch_triplet_loss += triplet_loss.item()
            epoch_temporal_loss += temporal_loss.item()
            epoch_total_loss += total_loss.item()
            
            if batch_idx % 50 == 0:
                logger.info(
                    f"Epoch {epoch}, Batch {batch_idx}: "
                    f"Triplet Loss: {triplet_loss.item():.4f}, "
                    f"Temporal Loss: {temporal_loss.item():.4f}, "
                    f"Total Loss: {total_loss.item():.4f}"
                )
        
        # Calculate average losses
        n_batches = len(dataloader)
        return {
            'triplet_loss': epoch_triplet_loss / n_batches,
            'temporal_loss': epoch_temporal_loss / n_batches,
            'total_loss': epoch_total_loss / n_batches
        }
    
    def apply_unsupervised_analysis(self, epoch: int) -> Dict:
        """Apply clustering analysis and pseudo-labeling every few epochs."""
        logger.info(f"🔍 Running unsupervised analysis at epoch {epoch}")
        
        results = self.unsupervised_pipeline.apply_unsupervised_techniques("")
        
        # Log results
        n_pseudo_labels = len(results.get('pseudo_labels', {}).get('pseudo_labels', {}))
        n_outliers = len(results.get('outliers', []))
        quality_score = results.get('quality_score', 0.0)
        
        logger.info(f"Analysis results: {n_pseudo_labels} pseudo-labels, "
                   f"{n_outliers} outliers, quality score: {quality_score:.3f}")
        
        # Log recommendations
        for rec in results.get('recommendations', []):
            logger.info(f"💡 Recommendation: {rec}")
        
        return results
    
    def evaluate_clustering_quality(self) -> float:
        """Evaluate current embedding quality using clustering metrics."""
        try:
            # Get all current embeddings
            crows = get_all_crows()
            if not crows:
                return 0.0
            
            all_embeddings = []
            for crow in crows:
                embeddings = get_crow_embeddings(crow['id'])
                if embeddings:
                    all_embeddings.extend([emb['embedding'] for emb in embeddings])
            
            if len(all_embeddings) < 10:
                return 0.0
            
            # Use clustering analyzer
            analyzer = CrowClusterAnalyzer()
            embeddings_array = np.array(all_embeddings)
            
            # Run clustering and get quality metrics
            labels, metrics = analyzer.cluster_crows(embeddings_array)
            
            # Extract quality score (silhouette or custom metric)
            quality_score = metrics.get('silhouette_score', 0.0)
            return max(0.0, min(1.0, quality_score))  # Normalize to [0, 1]
            
        except Exception as e:
            logger.warning(f"Could not evaluate clustering quality: {e}")
            return 0.0
    
    def train(self, dataloader, num_epochs: int, 
              crop_directories: List[str] = None) -> Dict:
        """
        Main training loop with unsupervised enhancements.
        
        Args:
            dataloader: Training data loader
            num_epochs: Number of epochs to train
            crop_directories: Directories for self-supervised pretraining
        
        Returns:
            Training statistics and results
        """
        logger.info(f"🚀 Starting enhanced training for {num_epochs} epochs")
        
        # Run self-supervised pretraining if requested
        pretraining_stats = {}
        if crop_directories and self.config.get('run_pretraining', True):
            pretraining_stats = self.run_self_supervised_pretraining(crop_directories)
        
        best_loss = float('inf')
        best_quality_score = 0.0
        
        for epoch in range(num_epochs):
            logger.info(f"📈 Starting epoch {epoch + 1}/{num_epochs}")
            
            # Train for one epoch
            epoch_stats = self.train_epoch(dataloader, epoch)
            
            # Update learning rate
            self.scheduler.step(epoch_stats['total_loss'])
            
            # Evaluate clustering quality
            quality_score = self.evaluate_clustering_quality()
            
            # Apply unsupervised analysis every 5 epochs
            unsupervised_results = {}
            if (epoch + 1) % 5 == 0:
                unsupervised_results = self.apply_unsupervised_analysis(epoch + 1)
            
            # Update training statistics
            self.training_stats['epochs'].append(epoch + 1)
            self.training_stats['triplet_losses'].append(epoch_stats['triplet_loss'])
            self.training_stats['temporal_losses'].append(epoch_stats['temporal_loss'])
            self.training_stats['total_losses'].append(epoch_stats['total_loss'])
            self.training_stats['clustering_scores'].append(quality_score)
            
            n_pseudo_labels = len(unsupervised_results.get('pseudo_labels', {}).get('pseudo_labels', {}))
            self.training_stats['pseudo_labels_generated'].append(n_pseudo_labels)
            
            # Save best model
            if epoch_stats['total_loss'] < best_loss or quality_score > best_quality_score:
                if epoch_stats['total_loss'] < best_loss:
                    best_loss = epoch_stats['total_loss']
                if quality_score > best_quality_score:
                    best_quality_score = quality_score
                
                self.save_checkpoint(epoch + 1, epoch_stats, quality_score)
            
            logger.info(
                f"Epoch {epoch + 1} completed: "
                f"Loss: {epoch_stats['total_loss']:.4f}, "
                f"Quality: {quality_score:.3f}, "
                f"Pseudo-labels: {n_pseudo_labels}"
            )
        
        logger.info("✅ Training completed!")
        
        return {
            'training_stats': self.training_stats,
            'pretraining_stats': pretraining_stats,
            'final_quality_score': best_quality_score,
            'best_loss': best_loss
        }
    
    def save_checkpoint(self, epoch: int, epoch_stats: Dict, quality_score: float):
        """Save model checkpoint with training statistics."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'epoch_stats': epoch_stats,
            'quality_score': quality_score,
            'training_stats': self.training_stats,
            'config': self.config
        }
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_path = f"unsupervised_model_epoch_{epoch}_{timestamp}.pth"
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"💾 Checkpoint saved: {checkpoint_path}")
        
        # Also save a "best" model
        torch.save(checkpoint, "best_unsupervised_model.pth")


def main():
    parser = argparse.ArgumentParser(description='Enhanced Training with Unsupervised Learning')
    parser.add_argument('--config', type=str, default='unsupervised_config.json',
                       help='Configuration file path')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--crop-dirs', nargs='+', 
                       help='Directories containing crop images for pretraining')
    parser.add_argument('--no-pretraining', action='store_true',
                       help='Skip self-supervised pretraining')
    
    args = parser.parse_args()
    
    # Load configuration
    if Path(args.config).exists():
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        # Default configuration
        config = {
            'embedding_dim': 512,
            'dropout_rate': 0.1,
            'margin': 1.0,
            'mining_type': 'adaptive',
            'learning_rate': 0.001,
            'weight_decay': 0.01,
            'temporal_weight': 0.1,
            'max_frames_gap': 5,
            'simclr_epochs': 50,
            'auto_label_confidence': 0.95,
            'run_pretraining': not args.no_pretraining
        }
        
        # Save default config
        with open(args.config, 'w') as f:
            json.dump(config, f, indent=2)
        logger.info(f"Created default configuration: {args.config}")
    
    # Initialize trainer
    trainer = UnsupervisedEnhancedTrainer(config)
    
    # Create dataset (you'll need to adapt this to your data structure)
    try:
        dataset = EnhancedTripletDataset(
            triplet_dir="triplets",  # Adjust path as needed
            transform=None,  # Add appropriate transforms
            augment=True
        )
        
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config.get('batch_size', 32),
            shuffle=True,
            num_workers=4
        )
        
        # Run training
        results = trainer.train(
            dataloader,
            args.epochs,
            crop_directories=args.crop_dirs
        )
        
        # Save final results
        results_path = f"training_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"📊 Training results saved: {results_path}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main() 