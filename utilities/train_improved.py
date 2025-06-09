#!/usr/bin/env python3
"""
Improved Training Script for Crow Identification
Incorporates all the fixes: better triplet mining, proper architecture, monitoring.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import time
from pathlib import Path
import logging
from datetime import datetime
import json
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
import warnings

# Import our modules
from models import CrowResNetEmbedder
from improved_dataset import ImprovedCrowTripletDataset
from improved_triplet_loss import ImprovedTripletLoss

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training.log')
    ]
)
logger = logging.getLogger(__name__)

class ImprovedTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Create output directories
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(exist_ok=True)
        self.checkpoint_dir = self.output_dir / 'checkpoints'
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Initialize model
        self.model = CrowResNetEmbedder(embedding_dim=config['embedding_dim']).to(self.device)
        logger.info(f"Created model with {config['embedding_dim']}D embeddings")
        
        # Initialize loss function with improved mining
        self.criterion = ImprovedTripletLoss(
            margin=config['margin'],
            mining_type=config['mining_type'],
            alpha=config.get('alpha', 0.2),
            beta=config.get('beta', 0.02)
        ).to(self.device)
        
        # Initialize optimizer with scheduler
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=config['learning_rate'],
            weight_decay=config.get('weight_decay', 1e-4)
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=config['epochs'],
            eta_min=config['learning_rate'] * 0.01
        )
        
        # Load existing weights if available (after optimizer creation)
        if config.get('resume_from') and os.path.exists(config['resume_from']):
            self._load_checkpoint(config['resume_from'])
        
        # Training state
        self.start_epoch = 0
        self.best_separability = 0.0
        self.training_history = {
            'train_loss': [],
            'eval_metrics': [],
            'learning_rates': [],
            'epochs': []
        }
        
        # Load datasets
        self._load_datasets()
        
    def _load_datasets(self):
        """Load training and validation datasets."""
        logger.info("Loading improved datasets...")
        
        # Training dataset with augmentation
        self.train_dataset = ImprovedCrowTripletDataset(
            self.config['crop_dir'], 
            split='train',
            transform_mode='augmented'  # Use augmentation for training
        )
        
        # Validation dataset without augmentation
        self.val_dataset = ImprovedCrowTripletDataset(
            self.config['crop_dir'],
            split='val',
            transform_mode='standard'  # No augmentation for validation
        )
        
        logger.info(f"Training dataset: {len(self.train_dataset)} samples")
        logger.info(f"Validation dataset: {len(self.val_dataset)} samples")
        logger.info(f"Training crows: {len(self.train_dataset.crow_to_imgs)}")
        logger.info(f"Validation crows: {len(self.val_dataset.crow_to_imgs)}")
        
        # Create data loaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=self.config.get('num_workers', 4),
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=self.config.get('num_workers', 4),
            pin_memory=True if torch.cuda.is_available() else False
        )
    
    def _load_checkpoint(self, checkpoint_path):
        """Load model checkpoint."""
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                # Full checkpoint with training state
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.start_epoch = checkpoint.get('epoch', 0)
                self.best_separability = checkpoint.get('best_separability', 0.0)
                self.training_history = checkpoint.get('training_history', self.training_history)
                logger.info(f"Loaded full checkpoint from epoch {self.start_epoch}")
            else:
                # Just model weights
                self.model.load_state_dict(checkpoint)
                logger.info("Loaded model weights only")
                
        except Exception as e:
            logger.warning(f"Could not load checkpoint: {e}")
    
    def _save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_separability': self.best_separability,
            'training_history': self.training_history,
            'config': self.config
        }
        
        # Save latest checkpoint
        checkpoint_path = self.checkpoint_dir / 'latest_checkpoint.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Save epoch-specific checkpoint
        if (epoch + 1) % self.config.get('save_every', 10) == 0:
            epoch_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch+1:03d}.pth'
            torch.save(checkpoint, epoch_path)
        
        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            # Also save just the model weights for easy loading
            model_path = self.output_dir / 'crow_resnet_triplet_improved.pth'
            torch.save(self.model.state_dict(), model_path)
            logger.info(f"Saved best model with separability: {self.best_separability:.4f}")
    
    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        # Update curriculum for improved dataset
        if hasattr(self.train_dataset, 'update_curriculum'):
            self.train_dataset.update_curriculum(epoch)
        
        # Progress bar
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.config["epochs"]}')
        
        for batch_idx, batch_data in enumerate(pbar):
            try:
                # Unpack batch
                if len(batch_data) == 3:
                    (anchor_imgs, pos_imgs, neg_imgs), (anchor_audio, pos_audio, neg_audio), labels = batch_data
                else:
                    # Fallback for different data format
                    anchor_imgs, pos_imgs, neg_imgs = batch_data[0]
                    labels = batch_data[-1]
                
                # Move to device
                anchor_imgs = anchor_imgs.to(self.device)
                pos_imgs = pos_imgs.to(self.device)
                neg_imgs = neg_imgs.to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                
                anchor_emb = self.model(anchor_imgs)
                pos_emb = self.model(pos_imgs)
                neg_emb = self.model(neg_imgs)
                
                # Concatenate embeddings and create labels for improved triplet loss
                all_embeddings = torch.cat([anchor_emb, pos_emb, neg_emb], dim=0)
                batch_size = anchor_emb.size(0)
                # Labels: anchor(0), positive(same as anchor), negative(different)
                all_labels = torch.cat([
                    labels,  # anchor labels
                    labels,  # positive labels (same as anchor)
                    labels + len(torch.unique(labels))  # negative labels (different)
                ], dim=0).to(self.device)
                
                # Compute improved triplet loss
                loss, stats = self.criterion(all_embeddings, all_labels)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                
                # Update metrics
                total_loss += loss.item()
                num_batches += 1
                
                # Update progress bar
                avg_loss = total_loss / num_batches
                pbar.set_postfix({
                    'Loss': f'{avg_loss:.4f}',
                    'LR': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
                })
                
            except Exception as e:
                logger.warning(f"Error in batch {batch_idx}: {e}")
                continue
        
        avg_loss = total_loss / max(num_batches, 1)
        current_lr = self.optimizer.param_groups[0]['lr']
        
        # Update scheduler
        self.scheduler.step()
        
        # Log metrics
        self.training_history['train_loss'].append(avg_loss)
        self.training_history['learning_rates'].append(current_lr)
        self.training_history['epochs'].append(epoch + 1)
        
        logger.info(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, LR={current_lr:.6f}")
        
        return avg_loss
    
    def evaluate_simple(self, epoch):
        """Simple evaluation using current model in memory."""
        self.model.eval()
        
        try:
            # Get a subset of validation data
            val_embeddings = []
            val_labels = []
            
            with torch.no_grad():
                for i, batch_data in enumerate(self.val_loader):
                    if i >= 10:  # Limit to first 10 batches for speed
                        break
                    
                    try:
                        if len(batch_data) == 3:
                            (anchor_imgs, _, _), _, labels = batch_data
                        else:
                            anchor_imgs, labels = batch_data[0][0], batch_data[-1]
                        
                        anchor_imgs = anchor_imgs.to(self.device)
                        embeddings = self.model(anchor_imgs)
                        
                        val_embeddings.append(embeddings.cpu().numpy())
                        val_labels.extend(labels)
                        
                    except Exception as e:
                        logger.warning(f"Error in validation batch {i}: {e}")
                        continue
            
            if len(val_embeddings) == 0:
                return 0.0
            
            # Compute basic metrics
            embeddings = np.vstack(val_embeddings)
            
            # Compute pairwise similarities
            from scipy.spatial.distance import cdist
            distances = cdist(embeddings, embeddings, metric='cosine')
            similarities = 1 - distances
            
            # Compute separability
            unique_labels = list(set(val_labels))
            same_crow_sims = []
            diff_crow_sims = []
            
            for i in range(len(val_labels)):
                for j in range(i+1, len(val_labels)):
                    sim = similarities[i, j]
                    if val_labels[i] == val_labels[j]:
                        same_crow_sims.append(sim)
                    else:
                        diff_crow_sims.append(sim)
            
            if len(same_crow_sims) > 0 and len(diff_crow_sims) > 0:
                separability = np.mean(same_crow_sims) - np.mean(diff_crow_sims)
                
                # Calculate precision, recall, and F1 score
                # Use a threshold to determine positive/negative predictions
                threshold = 0.5  # Similarity threshold for same crow prediction
                
                # True positives: same crow pairs with similarity >= threshold
                tp = sum(1 for sim in same_crow_sims if sim >= threshold)
                # False negatives: same crow pairs with similarity < threshold
                fn = sum(1 for sim in same_crow_sims if sim < threshold)
                # False positives: different crow pairs with similarity >= threshold
                fp = sum(1 for sim in diff_crow_sims if sim >= threshold)
                # True negatives: different crow pairs with similarity < threshold
                tn = sum(1 for sim in diff_crow_sims if sim < threshold)
                
                # Calculate metrics
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
                
                # Store evaluation metrics
                eval_data = {
                    'epoch': epoch + 1,
                    'separability': separability,
                    'same_crow_sim': np.mean(same_crow_sims),
                    'diff_crow_sim': np.mean(diff_crow_sims),
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1_score,
                    'threshold': threshold,
                    'tp': tp,
                    'fp': fp,
                    'tn': tn,
                    'fn': fn
                }
                self.training_history['eval_metrics'].append(eval_data)
                
                logger.info(f"Eval - Epoch {epoch+1}: Separability={separability:.3f}, "
                           f"Same={np.mean(same_crow_sims):.3f}, Diff={np.mean(diff_crow_sims):.3f}, "
                           f"P={precision:.3f}, R={recall:.3f}, F1={f1_score:.3f}")
                return separability
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Simple evaluation failed: {e}")
            return 0.0
        finally:
            self.model.train()
    
    def save_training_plots(self):
        """Save training progress plots."""
        if len(self.training_history['train_loss']) == 0:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Training loss
        axes[0, 0].plot(self.training_history['epochs'], self.training_history['train_loss'])
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True)
        
        # Learning rate
        axes[0, 1].plot(self.training_history['epochs'], self.training_history['learning_rates'])
        axes[0, 1].set_title('Learning Rate')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('LR')
        axes[0, 1].set_yscale('log')
        axes[0, 1].grid(True)
        
        # Evaluation metrics
        if len(self.training_history['eval_metrics']) > 0:
            eval_epochs = [m['epoch'] for m in self.training_history['eval_metrics']]
            separabilities = [m['separability'] for m in self.training_history['eval_metrics']]
            same_sims = [m['same_crow_sim'] for m in self.training_history['eval_metrics']]
            diff_sims = [m['diff_crow_sim'] for m in self.training_history['eval_metrics']]
            
            axes[1, 0].plot(eval_epochs, separabilities, 'b-', label='Separability')
            axes[1, 0].set_title('Separability')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Separability')
            axes[1, 0].grid(True)
            
            axes[1, 1].plot(eval_epochs, same_sims, 'g-', label='Same Crow')
            axes[1, 1].plot(eval_epochs, diff_sims, 'r-', label='Diff Crow')
            axes[1, 1].set_title('Similarity Scores')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Similarity')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'training_progress.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Saved training plots")
    
    def train(self):
        """Main training loop."""
        logger.info("Starting improved training...")
        logger.info(f"Configuration: {self.config}")
        
        try:
            for epoch in range(self.start_epoch, self.config['epochs']):
                # Train one epoch
                train_loss = self.train_epoch(epoch)
                
                # Evaluate periodically
                current_separability = 0.0
                if (epoch + 1) % self.config.get('eval_every', 5) == 0:
                    current_separability = self.evaluate_simple(epoch)
                
                # Save checkpoint
                is_best = current_separability > self.best_separability
                if is_best:
                    self.best_separability = current_separability
                
                self._save_checkpoint(epoch, is_best)
                
                # Save plots periodically
                if (epoch + 1) % self.config.get('plot_every', 10) == 0:
                    self.save_training_plots()
                
                # Early stopping check
                if self.config.get('early_stopping') and epoch > 20:
                    recent_separabilities = [m['separability'] for m in self.training_history['eval_metrics'][-5:]]
                    if len(recent_separabilities) >= 5 and all(s < 0.1 for s in recent_separabilities):
                        logger.warning("Early stopping: No improvement in separability")
                        break
        
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
        finally:
            # Final evaluation and save
            logger.info("Performing final evaluation...")
            final_separability = self.evaluate_simple(self.config['epochs'] - 1)
            
            # Save final model
            final_model_path = self.output_dir / 'crow_resnet_triplet_improved.pth'
            torch.save(self.model.state_dict(), final_model_path)
            logger.info(f"Saved final model to {final_model_path}")
            
            # Save training history
            history_path = self.output_dir / 'training_history.json'
            with open(history_path, 'w') as f:
                json.dump(self.training_history, f, indent=2)
            
            # Final plots
            self.save_training_plots()
            
            logger.info(f"Training completed! Best separability: {self.best_separability:.3f}")

def main():
    parser = argparse.ArgumentParser(description='Improved Crow Training')
    parser.add_argument('--crop-dir', default='crow_crops', help='Crop directory')
    parser.add_argument('--embedding-dim', type=int, default=256, help='Embedding dimension')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--margin', type=float, default=1.0, help='Triplet loss margin')
    parser.add_argument('--mining-type', default='adaptive', choices=['hard', 'semi_hard', 'adaptive', 'curriculum'])
    parser.add_argument('--output-dir', default='training_output_improved', help='Output directory')
    parser.add_argument('--resume-from', help='Resume from checkpoint')
    parser.add_argument('--eval-every', type=int, default=5, help='Evaluate every N epochs')
    parser.add_argument('--save-every', type=int, default=10, help='Save checkpoint every N epochs')
    parser.add_argument('--config', help='Load config from JSON file')
    
    args = parser.parse_args()
    
    # Load config from file if provided
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            file_config = json.load(f)
        config = file_config.get('training_config', {})
        logger.info(f"Loaded configuration from {args.config}")
    else:
        # Configuration from command line
        config = {
            'crop_dir': args.crop_dir,
            'embedding_dim': args.embedding_dim,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'margin': args.margin,
            'mining_type': args.mining_type,
            'output_dir': args.output_dir,
            'resume_from': args.resume_from,
            'eval_every': args.eval_every,
            'save_every': args.save_every,
            'weight_decay': 1e-4,
            'alpha': 0.2,
            'beta': 0.02,
            'num_workers': 4,
            'early_stopping': True,
            'plot_every': 10
        }
    
    # Create trainer and start training
    trainer = ImprovedTrainer(config)
    trainer.train()

if __name__ == '__main__':
    main() 