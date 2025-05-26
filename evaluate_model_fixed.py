#!/usr/bin/env python3
"""
Fixed Model Evaluation for Crow Identification
With improved error handling and debugging.
"""

import torch
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve, auc
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cdist
import json
import os
from pathlib import Path
import logging
from datetime import datetime
from collections import defaultdict
import warnings
import traceback

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    def __init__(self, model_path, crop_dir, audio_dir=None, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path
        self.crop_dir = crop_dir
        self.audio_dir = audio_dir
        
        logger.info(f"Initializing ModelEvaluator with device: {self.device}")
        logger.info(f"Model path: {model_path}")
        logger.info(f"Crop dir: {crop_dir}")
        
        # Check if model file exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Check if crop directory exists
        if not os.path.exists(crop_dir):
            raise FileNotFoundError(f"Crop directory not found: {crop_dir}")
        
        try:
            # Load model
            self.model = self._load_model()
            logger.info("Model loaded successfully")
            
            # Load dataset
            self.dataset = self._load_dataset()
            logger.info(f"Dataset loaded: {len(self.dataset)} samples, {len(self.dataset.crow_to_imgs)} crows")
            
        except Exception as e:
            logger.error(f"Error during initialization: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    def _load_model(self):
        """Load the trained model with better error handling."""
        try:
            logger.info("Attempting to load model...")
            
            # Try to load model state dict first to check compatibility
            state_dict = torch.load(self.model_path, map_location=self.device)
            logger.info("Model state dict loaded successfully")
            
            # Try multi-modal model first
            try:
                from models import CrowMultiModalEmbedder
                model = CrowMultiModalEmbedder(
                    visual_embed_dim=256,
                    audio_embed_dim=256,
                    final_embed_dim=512
                ).to(self.device)
                model.load_state_dict(state_dict)
                logger.info("Loaded as multi-modal model")
            except Exception as e:
                logger.info(f"Multi-modal model failed: {e}")
                # Fallback to visual-only model
                try:
                    from models import CrowResNetEmbedder
                    model = CrowResNetEmbedder(embedding_dim=512).to(self.device)
                    model.load_state_dict(state_dict)
                    logger.info("Loaded as visual-only ResNet model")
                except Exception as e2:
                    logger.error(f"Visual-only model also failed: {e2}")
                    # Try basic ResNet18 as last resort
                    import torchvision.models as models_tv
                    model = models_tv.resnet18(weights=None)
                    model.fc = torch.nn.Linear(model.fc.in_features, 512)
                    model = model.to(self.device)
                    
                    # Try to load compatible parts of state dict
                    model_dict = model.state_dict()
                    compatible_dict = {k: v for k, v in state_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
                    model_dict.update(compatible_dict)
                    model.load_state_dict(model_dict)
                    logger.info(f"Loaded as basic ResNet18 with {len(compatible_dict)} compatible layers")
            
            model.eval()
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    def _load_dataset(self):
        """Load dataset with better error handling."""
        try:
            from old_scripts.train_triplet_resnet import CrowTripletDataset
            
            # Create dataset
            dataset = CrowTripletDataset(self.crop_dir, self.audio_dir, split='val')
            
            if len(dataset) == 0:
                raise ValueError("Dataset is empty - no valid samples found")
            
            if len(dataset.crow_to_imgs) == 0:
                raise ValueError("No crow directories found in crop directory")
            
            logger.info(f"Found {len(dataset.crow_to_imgs)} crow directories")
            
            # Log some statistics
            sample_counts = [len(imgs) for imgs in dataset.crow_to_imgs.values()]
            logger.info(f"Sample count stats: min={min(sample_counts)}, max={max(sample_counts)}, mean={np.mean(sample_counts):.1f}")
            
            return dataset
            
        except Exception as e:
            logger.error(f"Failed to load dataset: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    def extract_embeddings_sample(self, max_samples=100):
        """Extract embeddings for a sample of data (for quick testing)."""
        embeddings = []
        labels = []
        image_paths = []
        
        sample_size = min(max_samples, len(self.dataset))
        logger.info(f"Extracting embeddings for {sample_size} samples...")
        
        with torch.no_grad():
            for i in range(sample_size):
                try:
                    # Get sample
                    sample_data = self.dataset[i]
                    
                    if len(sample_data) == 3:  # (imgs, audio, label)
                        (anchor_img, pos_img, neg_img), (anchor_audio, pos_audio, neg_audio), label = sample_data
                    else:
                        logger.warning(f"Unexpected sample format at index {i}: {len(sample_data)} elements")
                        continue
                    
                    # Move to device
                    anchor_img = anchor_img.unsqueeze(0).to(self.device)
                    
                    # Extract embedding
                    if hasattr(self.model, 'forward'):
                        # Check if model expects audio input
                        try:
                            if anchor_audio is not None:
                                anchor_audio = anchor_audio.unsqueeze(0).to(self.device)
                                emb = self.model(anchor_img, anchor_audio)
                            else:
                                emb = self.model(anchor_img)
                        except TypeError:
                            # Model doesn't accept audio input
                            emb = self.model(anchor_img)
                    else:
                        emb = self.model(anchor_img)
                    
                    # Handle different output formats
                    if isinstance(emb, dict):
                        emb = emb['embedding']  # Multi-task model
                    
                    embeddings.append(emb.cpu().numpy().flatten())
                    labels.append(label)
                    
                    if hasattr(self.dataset, 'samples') and i < len(self.dataset.samples):
                        image_paths.append(self.dataset.samples[i][0])
                    else:
                        image_paths.append(f"sample_{i}")
                    
                except Exception as e:
                    logger.warning(f"Failed to process sample {i}: {e}")
                    continue
        
        if len(embeddings) == 0:
            raise ValueError("No embeddings could be extracted")
        
        logger.info(f"Successfully extracted {len(embeddings)} embeddings")
        return np.array(embeddings), labels, image_paths
    
    def compute_basic_metrics(self, embeddings, labels):
        """Compute basic identification metrics."""
        unique_labels = list(set(labels))
        n_crows = len(unique_labels)
        
        logger.info(f"Computing metrics for {len(embeddings)} embeddings, {n_crows} unique crows")
        
        # Compute pairwise distances
        distances = cdist(embeddings, embeddings, metric='cosine')
        similarities = 1 - distances
        
        # Generate ground truth matrix
        label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        gt_matrix = np.array([[label_to_idx[labels[i]] == label_to_idx[labels[j]] 
                              for j in range(len(labels))] for i in range(len(labels))])
        
        # Exclude diagonal (self-similarity)
        mask = ~np.eye(len(labels), dtype=bool)
        similarities_flat = similarities[mask]
        gt_flat = gt_matrix[mask]
        
        # Compute basic metrics at threshold 0.5
        threshold = 0.5
        predictions = similarities_flat >= threshold
        
        tp = np.sum(predictions & gt_flat)
        fp = np.sum(predictions & ~gt_flat)
        fn = np.sum(~predictions & gt_flat)
        tn = np.sum(~predictions & ~gt_flat)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / (tp + fp + fn + tn)
        
        # Separability analysis
        same_crow_sims = similarities_flat[gt_flat]
        diff_crow_sims = similarities_flat[~gt_flat]
        
        separability_ratio = np.mean(diff_crow_sims) / np.mean(same_crow_sims) if np.mean(same_crow_sims) > 0 else 0
        
        return {
            'n_crows': n_crows,
            'n_samples': len(embeddings),
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'accuracy': accuracy,
            'same_crow_sim_mean': np.mean(same_crow_sims),
            'same_crow_sim_std': np.std(same_crow_sims),
            'diff_crow_sim_mean': np.mean(diff_crow_sims),
            'diff_crow_sim_std': np.std(diff_crow_sims),
            'separability_ratio': separability_ratio
        }
    
    def quick_evaluate(self, max_samples=100):
        """Run a quick evaluation on a sample of data."""
        logger.info("Starting quick evaluation...")
        
        try:
            # Extract embeddings for sample
            embeddings, labels, image_paths = self.extract_embeddings_sample(max_samples)
            
            # Compute basic metrics
            metrics = self.compute_basic_metrics(embeddings, labels)
            
            # Print results
            logger.info("="*60)
            logger.info("QUICK EVALUATION RESULTS")
            logger.info("="*60)
            logger.info(f"Dataset: {metrics['n_samples']} samples, {metrics['n_crows']} crows")
            logger.info(f"Precision: {metrics['precision']:.3f}")
            logger.info(f"Recall: {metrics['recall']:.3f}")
            logger.info(f"F1 Score: {metrics['f1_score']:.3f}")
            logger.info(f"Accuracy: {metrics['accuracy']:.3f}")
            logger.info(f"Same Crow Similarity: {metrics['same_crow_sim_mean']:.3f} ± {metrics['same_crow_sim_std']:.3f}")
            logger.info(f"Different Crow Similarity: {metrics['diff_crow_sim_mean']:.3f} ± {metrics['diff_crow_sim_std']:.3f}")
            logger.info(f"Separability Ratio: {metrics['separability_ratio']:.3f}")
            logger.info("="*60)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error during evaluation: {str(e)}")
            logger.error(traceback.format_exc())
            raise

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate crow identification model (fixed version)')
    parser.add_argument('--model-path', required=True, help='Path to trained model')
    parser.add_argument('--crop-dir', default='crow_crops', help='Directory with crow crops')
    parser.add_argument('--audio-dir', default=None, help='Directory with audio files')
    parser.add_argument('--max-samples', type=int, default=500, help='Maximum samples for quick evaluation')
    
    args = parser.parse_args()
    
    try:
        # Run evaluation
        evaluator = ModelEvaluator(args.model_path, args.crop_dir, args.audio_dir)
        results = evaluator.quick_evaluate(args.max_samples)
        
        print(f"\nQuick evaluation complete.")
        
    except Exception as e:
        print(f"Evaluation failed: {str(e)}")
        print("Full traceback:")
        print(traceback.format_exc())

if __name__ == '__main__':
    main() 