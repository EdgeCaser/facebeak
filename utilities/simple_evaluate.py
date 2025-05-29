#!/usr/bin/env python3
"""
Simple Model Evaluation with Correct Architecture
Loads the model with 128-dimensional embeddings.
"""

import torch
import torch.nn as nn
import numpy as np
from scipy.spatial.distance import cdist
import os
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CrowResNet128(nn.Module):
    """ResNet model with 128-dimensional embeddings (matching your trained model)."""
    def __init__(self):
        super(CrowResNet128, self).__init__()
        import torchvision.models as models
        
        # Use ResNet18 backbone
        resnet = models.resnet18(weights=None)
        
        # Remove the last fully connected layer
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        
        # Add custom fully connected layer for 128-dim embeddings
        self.fc = nn.Linear(512, 128)
        
    def forward(self, x):
        # Extract features
        features = self.feature_extractor(x)
        features = torch.flatten(features, 1)
        
        # Get embeddings
        embeddings = self.fc(features)
        
        return embeddings

def load_model_correct(model_path, device):
    """Load model with correct architecture."""
    model = CrowResNet128().to(device)
    state_dict = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(state_dict)
    model.eval()
    return model

def load_sample_data(crop_dir, max_crows=20, max_samples_per_crow=10):
    """Load a sample of data for quick evaluation."""
    from PIL import Image
    import torchvision.transforms as transforms
    
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    images = []
    labels = []
    
    crop_path = Path(crop_dir)
    crow_dirs = [d for d in crop_path.iterdir() if d.is_dir()][:max_crows]
    
    for crow_dir in crow_dirs:
        crow_id = crow_dir.name
        image_files = list(crow_dir.glob("*.jpg"))[:max_samples_per_crow]
        
        for img_file in image_files:
            try:
                img = Image.open(img_file).convert('RGB')
                img_tensor = transform(img)
                images.append(img_tensor)
                labels.append(crow_id)
            except Exception as e:
                logger.warning(f"Failed to load {img_file}: {e}")
                continue
    
    return torch.stack(images), labels

def evaluate_model(model_path, crop_dir, device=None):
    """Evaluate the model on sample data."""
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load model
    model = load_model_correct(model_path, device)
    logger.info("Model loaded successfully")
    
    # Load sample data
    images, labels = load_sample_data(crop_dir)
    logger.info(f"Loaded {len(images)} images from {len(set(labels))} crows")
    
    # Extract embeddings
    embeddings = []
    batch_size = 16
    
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size].to(device)
            emb = model(batch)
            embeddings.append(emb.cpu().numpy())
    
    embeddings = np.vstack(embeddings)
    logger.info(f"Extracted embeddings: {embeddings.shape}")
    
    # Compute basic metrics
    unique_labels = list(set(labels))
    n_crows = len(unique_labels)
    
    # Compute pairwise cosine distances
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
    
    # Compute metrics at threshold 0.5
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
    
    # Print results
    print("\n" + "="*60)
    print("FACEBEAK MODEL EVALUATION RESULTS")
    print("="*60)
    print(f"Dataset: {len(embeddings)} samples, {n_crows} unique crows")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    print(f"")
    print(f"Performance Metrics @ threshold {threshold}:")
    print(f"  Precision:  {precision:.3f}")
    print(f"  Recall:     {recall:.3f}")
    print(f"  F1 Score:   {f1:.3f}")
    print(f"  Accuracy:   {accuracy:.3f}")
    print(f"")
    print(f"Similarity Analysis:")
    print(f"  Same Crow:     {np.mean(same_crow_sims):.3f} ± {np.std(same_crow_sims):.3f}")
    print(f"  Different Crow: {np.mean(diff_crow_sims):.3f} ± {np.std(diff_crow_sims):.3f}")
    print(f"  Separability:   {np.mean(same_crow_sims) - np.mean(diff_crow_sims):.3f}")
    print("="*60)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'accuracy': accuracy,
        'same_crow_sim_mean': np.mean(same_crow_sims),
        'diff_crow_sim_mean': np.mean(diff_crow_sims),
        'separability': np.mean(same_crow_sims) - np.mean(diff_crow_sims),
        'n_samples': len(embeddings),
        'n_crows': n_crows
    }

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Simple model evaluation')
    parser.add_argument('--model-path', default='crow_resnet_triplet.pth', help='Path to model')
    parser.add_argument('--crop-dir', default='crow_crops', help='Crop directory')
    
    args = parser.parse_args()
    
    try:
        results = evaluate_model(args.model_path, args.crop_dir)
        print(f"\nEvaluation complete!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main() 