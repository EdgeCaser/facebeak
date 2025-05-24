#!/usr/bin/env python3
"""
Comprehensive Model Evaluation for Crow Identification
Implements proper validation metrics for triplet loss models.
"""

import torch
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve, auc, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cdist
import json
import os
from pathlib import Path
from models import CrowResNetEmbedder, CrowMultiModalEmbedder
from train_triplet_resnet import CrowTripletDataset
import logging
from datetime import datetime
from collections import defaultdict
import warnings

warnings.filterwarnings('ignore')

class ModelEvaluator:
    def __init__(self, model_path, crop_dir, audio_dir=None, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path
        self.crop_dir = crop_dir
        self.audio_dir = audio_dir
        
        # Load model
        self.model = self._load_model()
        
        # Load dataset
        self.dataset = CrowTripletDataset(crop_dir, audio_dir, split='val')
        
        logging.info(f"Loaded model from {model_path}")
        logging.info(f"Dataset: {len(self.dataset)} samples, {len(self.dataset.crow_to_imgs)} crows")
        
    def _load_model(self):
        """Load the trained model."""
        try:
            # Try multi-modal model first
            model = CrowMultiModalEmbedder(
                visual_embed_dim=256,
                audio_embed_dim=256,
                final_embed_dim=512
            ).to(self.device)
            model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        except:
            # Fallback to visual-only model
            model = CrowResNetEmbedder(embedding_dim=512).to(self.device)
            model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        
        model.eval()
        return model
    
    def extract_all_embeddings(self):
        """Extract embeddings for all samples."""
        embeddings = []
        labels = []
        image_paths = []
        
        with torch.no_grad():
            for i in range(len(self.dataset)):
                try:
                    # Get sample
                    (anchor_img, pos_img, neg_img), (anchor_audio, pos_audio, neg_audio), label = self.dataset[i]
                    
                    # Move to device
                    anchor_img = anchor_img.unsqueeze(0).to(self.device)
                    if anchor_audio is not None:
                        anchor_audio = anchor_audio.unsqueeze(0).to(self.device)
                    
                    # Extract embedding
                    if hasattr(self.model, 'forward'):
                        if anchor_audio is not None:
                            emb = self.model(anchor_img, anchor_audio)
                        else:
                            emb = self.model(anchor_img)
                    else:
                        emb = self.model(anchor_img)
                    
                    embeddings.append(emb.cpu().numpy().flatten())
                    labels.append(label)
                    image_paths.append(self.dataset.samples[i][0])  # image path
                    
                except Exception as e:
                    logging.warning(f"Failed to process sample {i}: {e}")
                    continue
        
        return np.array(embeddings), labels, image_paths
    
    def compute_identification_metrics(self, embeddings, labels, threshold_range=(0.1, 0.9, 50)):
        """Compute identification metrics across similarity thresholds."""
        unique_labels = list(set(labels))
        n_crows = len(unique_labels)
        
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
        
        # Compute metrics across thresholds
        thresholds = np.linspace(*threshold_range)
        precision_scores = []
        recall_scores = []
        f1_scores = []
        
        for threshold in thresholds:
            predictions = similarities_flat >= threshold
            
            tp = np.sum(predictions & gt_flat)
            fp = np.sum(predictions & ~gt_flat)
            fn = np.sum(~predictions & gt_flat)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            precision_scores.append(precision)
            recall_scores.append(recall)
            f1_scores.append(f1)
        
        return {
            'thresholds': thresholds,
            'precision': precision_scores,
            'recall': recall_scores,
            'f1': f1_scores,
            'similarities': similarities_flat,
            'ground_truth': gt_flat,
            'n_crows': n_crows
        }
    
    def compute_retrieval_metrics(self, embeddings, labels, k_values=[1, 3, 5, 10]):
        """Compute top-K retrieval accuracy."""
        unique_labels = list(set(labels))
        label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        
        # Compute distances
        distances = cdist(embeddings, embeddings, metric='cosine')
        
        retrieval_metrics = {}
        
        for k in k_values:
            correct_retrievals = 0
            total_queries = 0
            
            for i, query_label in enumerate(labels):
                # Get distances for this query (excluding self)
                query_distances = distances[i].copy()
                query_distances[i] = np.inf  # Exclude self
                
                # Get top-k nearest neighbors
                top_k_indices = np.argsort(query_distances)[:k]
                top_k_labels = [labels[idx] for idx in top_k_indices]
                
                # Check if any of top-k matches query label
                if query_label in top_k_labels:
                    correct_retrievals += 1
                total_queries += 1
            
            retrieval_metrics[f'top_{k}_accuracy'] = correct_retrievals / total_queries
        
        return retrieval_metrics
    
    def analyze_embedding_separability(self, embeddings, labels):
        """Analyze how well embeddings separate different crows."""
        unique_labels = list(set(labels))
        
        # Compute intra-class and inter-class distances
        intra_class_dists = []
        inter_class_dists = []
        
        for label in unique_labels:
            label_indices = [i for i, l in enumerate(labels) if l == label]
            other_indices = [i for i, l in enumerate(labels) if l != label]
            
            if len(label_indices) > 1:
                # Intra-class distances
                for i in range(len(label_indices)):
                    for j in range(i + 1, len(label_indices)):
                        dist = np.linalg.norm(embeddings[label_indices[i]] - embeddings[label_indices[j]])
                        intra_class_dists.append(dist)
            
            # Inter-class distances
            for i in label_indices:
                for j in other_indices:
                    dist = np.linalg.norm(embeddings[i] - embeddings[j])
                    inter_class_dists.append(dist)
        
        return {
            'intra_class_mean': np.mean(intra_class_dists),
            'intra_class_std': np.std(intra_class_dists),
            'inter_class_mean': np.mean(inter_class_dists),
            'inter_class_std': np.std(inter_class_dists),
            'separability_ratio': np.mean(inter_class_dists) / np.mean(intra_class_dists),
            'intra_class_dists': intra_class_dists,
            'inter_class_dists': inter_class_dists
        }
    
    def cross_validation_evaluation(self, n_folds=5):
        """Perform cross-validation evaluation."""
        embeddings, labels, _ = self.extract_all_embeddings()
        
        # Convert labels to numeric
        unique_labels = list(set(labels))
        label_to_num = {label: i for i, label in enumerate(unique_labels)}
        numeric_labels = np.array([label_to_num[label] for label in labels])
        
        # Stratified K-fold
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        cv_results = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(embeddings, numeric_labels)):
            train_emb, val_emb = embeddings[train_idx], embeddings[val_idx]
            train_labels, val_labels = numeric_labels[train_idx], numeric_labels[val_idx]
            
            # Train KNN classifier
            knn = KNeighborsClassifier(n_neighbors=3, metric='cosine')
            knn.fit(train_emb, train_labels)
            
            # Predict
            predictions = knn.predict(val_emb)
            accuracy = np.mean(predictions == val_labels)
            
            cv_results.append({
                'fold': fold + 1,
                'accuracy': accuracy,
                'n_train': len(train_idx),
                'n_val': len(val_idx)
            })
        
        return cv_results
    
    def plot_evaluation_results(self, results, output_dir):
        """Create comprehensive evaluation plots."""
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Precision-Recall Curve
        plt.figure(figsize=(12, 10))
        
        plt.subplot(2, 2, 1)
        plt.plot(results['identification']['recall'], results['identification']['precision'])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.grid(True)
        
        # 2. ROC Curve
        plt.subplot(2, 2, 2)
        fpr, tpr, _ = roc_curve(results['identification']['ground_truth'], 
                               results['identification']['similarities'])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True)
        
        # 3. Similarity Distribution
        plt.subplot(2, 2, 3)
        same_crow_sims = results['identification']['similarities'][results['identification']['ground_truth']]
        diff_crow_sims = results['identification']['similarities'][~results['identification']['ground_truth']]
        
        plt.hist(same_crow_sims, bins=50, alpha=0.7, label='Same Crow', density=True)
        plt.hist(diff_crow_sims, bins=50, alpha=0.7, label='Different Crow', density=True)
        plt.xlabel('Cosine Similarity')
        plt.ylabel('Density')
        plt.title('Similarity Distribution')
        plt.legend()
        
        # 4. Top-K Accuracy
        plt.subplot(2, 2, 4)
        k_values = [int(k.split('_')[1]) for k in results['retrieval'].keys()]
        accuracies = list(results['retrieval'].values())
        plt.bar(k_values, accuracies)
        plt.xlabel('K')
        plt.ylabel('Top-K Accuracy')
        plt.title('Retrieval Performance')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'evaluation_results.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 5. Embedding Separability
        plt.figure(figsize=(10, 6))
        plt.hist(results['separability']['intra_class_dists'], bins=50, alpha=0.7, 
                label='Intra-class', density=True)
        plt.hist(results['separability']['inter_class_dists'], bins=50, alpha=0.7, 
                label='Inter-class', density=True)
        plt.xlabel('Euclidean Distance')
        plt.ylabel('Density')
        plt.title('Embedding Distance Distribution')
        plt.legend()
        plt.savefig(os.path.join(output_dir, 'separability_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def evaluate(self, output_dir='evaluation_results'):
        """Run complete evaluation."""
        logging.info("Starting comprehensive model evaluation...")
        
        # Extract embeddings
        embeddings, labels, image_paths = self.extract_all_embeddings()
        
        # Compute all metrics
        results = {
            'model_path': self.model_path,
            'dataset_size': len(embeddings),
            'n_crows': len(set(labels)),
            'timestamp': datetime.now().isoformat()
        }
        
        # Identification metrics
        results['identification'] = self.compute_identification_metrics(embeddings, labels)
        
        # Retrieval metrics
        results['retrieval'] = self.compute_retrieval_metrics(embeddings, labels)
        
        # Separability analysis
        results['separability'] = self.analyze_embedding_separability(embeddings, labels)
        
        # Cross-validation
        results['cross_validation'] = self.cross_validation_evaluation()
        
        # Summary metrics
        best_f1_idx = np.argmax(results['identification']['f1'])
        results['summary'] = {
            'best_f1_score': results['identification']['f1'][best_f1_idx],
            'best_threshold': results['identification']['thresholds'][best_f1_idx],
            'top_1_accuracy': results['retrieval']['top_1_accuracy'],
            'top_5_accuracy': results['retrieval']['top_5_accuracy'],
            'separability_ratio': results['separability']['separability_ratio'],
            'cv_mean_accuracy': np.mean([cv['accuracy'] for cv in results['cross_validation']]),
            'cv_std_accuracy': np.std([cv['accuracy'] for cv in results['cross_validation']])
        }
        
        # Save results
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, 'evaluation_results.json'), 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            json_results = self._convert_numpy_to_list(results)
            json.dump(json_results, f, indent=2)
        
        # Create plots
        self.plot_evaluation_results(results, output_dir)
        
        # Print summary
        self._print_summary(results['summary'])
        
        return results
    
    def _convert_numpy_to_list(self, obj):
        """Convert numpy arrays to lists for JSON serialization."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._convert_numpy_to_list(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_to_list(item) for item in obj]
        else:
            return obj
    
    def _print_summary(self, summary):
        """Print evaluation summary."""
        print("\n" + "="*60)
        print("MODEL EVALUATION SUMMARY")
        print("="*60)
        print(f"Best F1 Score: {summary['best_f1_score']:.3f}")
        print(f"Best Threshold: {summary['best_threshold']:.3f}")
        print(f"Top-1 Accuracy: {summary['top_1_accuracy']:.3f}")
        print(f"Top-5 Accuracy: {summary['top_5_accuracy']:.3f}")
        print(f"Separability Ratio: {summary['separability_ratio']:.3f}")
        print(f"CV Accuracy: {summary['cv_mean_accuracy']:.3f} ± {summary['cv_std_accuracy']:.3f}")
        print("="*60)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate crow identification model')
    parser.add_argument('--model-path', required=True, help='Path to trained model')
    parser.add_argument('--crop-dir', default='crow_crops', help='Directory with crow crops')
    parser.add_argument('--audio-dir', default=None, help='Directory with audio files')
    parser.add_argument('--output-dir', default='evaluation_results', help='Output directory')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Run evaluation
    evaluator = ModelEvaluator(args.model_path, args.crop_dir, args.audio_dir)
    results = evaluator.evaluate(args.output_dir)
    
    print(f"\nEvaluation complete. Results saved to {args.output_dir}")

if __name__ == '__main__':
    main() 