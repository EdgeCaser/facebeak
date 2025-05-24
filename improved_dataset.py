#!/usr/bin/env python3
"""
Improved Dataset for Crow Triplet Training
Includes better augmentation, curriculum learning, and data balancing.
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
from PIL import Image
import torchvision.transforms as transforms
from pathlib import Path
import random
import logging
from collections import defaultdict
import json

logger = logging.getLogger(__name__)

class ImprovedCrowTripletDataset(Dataset):
    def __init__(self, crop_dir, split='train', transform_mode='standard', 
                 min_samples_per_crow=5, max_samples_per_crow=200,
                 curriculum_epoch=0, max_curriculum_epochs=20):
        """
        Improved Crow Triplet Dataset with curriculum learning and balancing.
        
        Args:
            crop_dir: Directory containing crow crops
            split: 'train' or 'val'
            transform_mode: 'standard', 'augmented', or 'heavy'
            min_samples_per_crow: Minimum samples required per crow
            max_samples_per_crow: Maximum samples to use per crow
            curriculum_epoch: Current epoch for curriculum learning
            max_curriculum_epochs: Maximum epochs for curriculum phase
        """
        self.crop_dir = Path(crop_dir)
        self.split = split
        self.transform_mode = transform_mode
        self.min_samples_per_crow = min_samples_per_crow
        self.max_samples_per_crow = max_samples_per_crow
        self.curriculum_epoch = curriculum_epoch
        self.max_curriculum_epochs = max_curriculum_epochs
        
        # Setup transforms
        self._setup_transforms()
        
        # Load and balance dataset
        self._load_and_balance_dataset()
        
        # Setup curriculum learning
        self._setup_curriculum()
        
        logger.info(f"Dataset {split}: {len(self.samples)} samples from {len(self.crow_to_imgs)} crows")
        
    def _setup_transforms(self):
        """Setup data transforms based on mode."""
        # Base transforms
        base_transforms = [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
        
        if self.transform_mode == 'standard':
            self.transform = transforms.Compose(base_transforms)
            
        elif self.transform_mode == 'augmented':
            augmented_transforms = [
                transforms.Resize((256, 256)),
                transforms.RandomCrop((224, 224)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                transforms.RandomErasing(p=0.1, scale=(0.02, 0.1))
            ]
            self.transform = transforms.Compose(augmented_transforms)
            
        elif self.transform_mode == 'heavy':
            heavy_transforms = [
                transforms.Resize((256, 256)),
                transforms.RandomCrop((224, 224)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15),
                transforms.RandomRotation(15),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                transforms.RandomErasing(p=0.2, scale=(0.02, 0.15))
            ]
            self.transform = transforms.Compose(heavy_transforms)
    
    def _load_and_balance_dataset(self):
        """Load and balance the dataset."""
        logger.info("Loading and balancing dataset...")
        
        # Find all crow directories
        crow_dirs = [d for d in self.crop_dir.iterdir() if d.is_dir()]
        crow_dirs.sort(key=lambda x: x.name)
        
        # Split into train/val (80/20)
        split_idx = int(len(crow_dirs) * 0.8)
        if self.split == 'train':
            selected_dirs = crow_dirs[:split_idx]
        else:
            selected_dirs = crow_dirs[split_idx:]
        
        # Load images for each crow
        self.crow_to_imgs = {}
        for crow_dir in selected_dirs:
            crow_id = crow_dir.name
            img_files = list(crow_dir.glob("*.jpg")) + list(crow_dir.glob("*.png"))
            
            if len(img_files) >= self.min_samples_per_crow:
                # Balance the number of samples
                if len(img_files) > self.max_samples_per_crow:
                    # Randomly sample max_samples_per_crow images
                    img_files = random.sample(img_files, self.max_samples_per_crow)
                
                self.crow_to_imgs[crow_id] = img_files
        
        # Filter out crows with too few samples
        filtered_crows = {k: v for k, v in self.crow_to_imgs.items() 
                         if len(v) >= self.min_samples_per_crow}
        self.crow_to_imgs = filtered_crows
        
        # Create sample list
        self.samples = []
        for crow_id, img_files in self.crow_to_imgs.items():
            for img_file in img_files:
                self.samples.append((img_file, crow_id))
        
        # Balance dataset by oversampling minority classes
        self._balance_classes()
        
        logger.info(f"Loaded {len(self.samples)} samples from {len(self.crow_to_imgs)} crows")
        
        # Log class distribution
        class_counts = defaultdict(int)
        for _, crow_id in self.samples:
            class_counts[crow_id] += 1
        
        logger.info(f"Class distribution: min={min(class_counts.values())}, "
                   f"max={max(class_counts.values())}, "
                   f"mean={np.mean(list(class_counts.values())):.1f}")
    
    def _balance_classes(self):
        """Balance classes by oversampling."""
        # Count samples per crow
        crow_counts = defaultdict(int)
        for _, crow_id in self.samples:
            crow_counts[crow_id] += 1
        
        # Find target count (75th percentile)
        counts = list(crow_counts.values())
        target_count = int(np.percentile(counts, 75))
        
        # Oversample minority classes
        balanced_samples = []
        for crow_id, img_files in self.crow_to_imgs.items():
            current_count = crow_counts[crow_id]
            
            if current_count < target_count:
                # Oversample by repeating images
                multiplier = target_count // current_count
                remainder = target_count % current_count
                
                # Add full repetitions
                for _ in range(multiplier):
                    for img_file in img_files:
                        balanced_samples.append((img_file, crow_id))
                
                # Add remainder
                if remainder > 0:
                    remaining_files = random.sample(img_files, remainder)
                    for img_file in remaining_files:
                        balanced_samples.append((img_file, crow_id))
            else:
                # Keep original samples
                for img_file in img_files:
                    balanced_samples.append((img_file, crow_id))
        
        self.samples = balanced_samples
        logger.info(f"Balanced dataset: {len(self.samples)} samples")
    
    def _setup_curriculum(self):
        """Setup curriculum learning."""
        if self.curriculum_epoch < self.max_curriculum_epochs:
            # During curriculum phase, use easier examples
            difficulty_ratio = min(1.0, self.curriculum_epoch / self.max_curriculum_epochs)
            
            # Sort samples by some difficulty metric (e.g., file size as proxy for image quality)
            sample_difficulties = []
            for img_path, crow_id in self.samples:
                try:
                    file_size = img_path.stat().st_size
                    sample_difficulties.append((img_path, crow_id, file_size))
                except:
                    sample_difficulties.append((img_path, crow_id, 0))
            
            # Sort by file size (larger = better quality = easier)
            sample_difficulties.sort(key=lambda x: x[2], reverse=True)
            
            # Use only top percentage based on curriculum progress
            num_samples = int(len(sample_difficulties) * (0.3 + 0.7 * difficulty_ratio))
            self.samples = [(path, crow_id) for path, crow_id, _ in sample_difficulties[:num_samples]]
            
            logger.info(f"Curriculum learning: using {num_samples}/{len(sample_difficulties)} samples "
                       f"(difficulty ratio: {difficulty_ratio:.2f})")
    
    def update_curriculum(self, epoch):
        """Update curriculum for new epoch."""
        if epoch != self.curriculum_epoch and epoch < self.max_curriculum_epochs:
            self.curriculum_epoch = epoch
            self._setup_curriculum()
    
    def __len__(self):
        return len(self.samples)
    
    def _load_image(self, img_path):
        """Load and preprocess image."""
        try:
            # Load image
            img = Image.open(img_path).convert('RGB')
            
            # Apply transforms
            if self.transform:
                img = self.transform(img)
            
            return img
        except Exception as e:
            logger.warning(f"Failed to load image {img_path}: {e}")
            # Return a random noise image as fallback
            return torch.randn(3, 224, 224)
    
    def _get_positive_sample(self, anchor_crow_id, anchor_path):
        """Get a positive sample (same crow, different image)."""
        possible_positives = [path for path in self.crow_to_imgs[anchor_crow_id] 
                            if path != anchor_path]
        
        if not possible_positives:
            # If no other images available, use the same image
            return anchor_path
        
        return random.choice(possible_positives)
    
    def _get_negative_sample(self, anchor_crow_id):
        """Get a negative sample (different crow)."""
        possible_crows = [crow_id for crow_id in self.crow_to_imgs.keys() 
                         if crow_id != anchor_crow_id]
        
        if not possible_crows:
            # Should not happen, but fallback to anchor crow
            neg_crow_id = anchor_crow_id
        else:
            neg_crow_id = random.choice(possible_crows)
        
        return random.choice(self.crow_to_imgs[neg_crow_id])
    
    def _get_hard_negative_sample(self, anchor_crow_id, anchor_embedding=None):
        """Get a hard negative sample using curriculum learning."""
        # For now, just return a random negative
        # TODO: Implement hard negative mining using pre-computed embeddings
        return self._get_negative_sample(anchor_crow_id)
    
    def __getitem__(self, idx):
        """Get a triplet sample."""
        # Get anchor
        anchor_path, anchor_crow_id = self.samples[idx]
        anchor_img = self._load_image(anchor_path)
        
        # Get positive
        pos_path = self._get_positive_sample(anchor_crow_id, anchor_path)
        pos_img = self._load_image(pos_path)
        
        # Get negative
        neg_path = self._get_negative_sample(anchor_crow_id)
        neg_img = self._load_image(neg_path)
        
        # Return triplet
        imgs = (anchor_img, pos_img, neg_img)
        
        # For compatibility with audio datasets, return None for audio
        audio = (None, None, None)
        
        # Return label (crow_id)
        label = anchor_crow_id
        
        return imgs, audio, label

class DatasetStats:
    """Utility class for dataset statistics."""
    
    @staticmethod
    def analyze_dataset(crop_dir):
        """Analyze dataset and provide statistics."""
        crop_path = Path(crop_dir)
        crow_dirs = [d for d in crop_path.iterdir() if d.is_dir()]
        
        stats = {
            'total_crows': len(crow_dirs),
            'total_images': 0,
            'images_per_crow': [],
            'image_sizes': [],
            'file_sizes': []
        }
        
        for crow_dir in crow_dirs:
            img_files = list(crow_dir.glob("*.jpg")) + list(crow_dir.glob("*.png"))
            stats['total_images'] += len(img_files)
            stats['images_per_crow'].append(len(img_files))
            
            for img_file in img_files[:10]:  # Sample first 10 for size analysis
                try:
                    img = Image.open(img_file)
                    stats['image_sizes'].append(img.size)
                    stats['file_sizes'].append(img_file.stat().st_size)
                except:
                    continue
        
        # Compute statistics
        if stats['images_per_crow']:
            stats['images_per_crow_stats'] = {
                'min': min(stats['images_per_crow']),
                'max': max(stats['images_per_crow']),
                'mean': np.mean(stats['images_per_crow']),
                'median': np.median(stats['images_per_crow']),
                'std': np.std(stats['images_per_crow'])
            }
        
        if stats['file_sizes']:
            stats['file_size_stats'] = {
                'min': min(stats['file_sizes']),
                'max': max(stats['file_sizes']),
                'mean': np.mean(stats['file_sizes']),
                'median': np.median(stats['file_sizes'])
            }
        
        return stats
    
    @staticmethod
    def recommend_training_params(crop_dir):
        """Recommend training parameters based on dataset analysis."""
        stats = DatasetStats.analyze_dataset(crop_dir)
        
        recommendations = {}
        
        # Batch size based on total images
        if stats['total_images'] < 1000:
            recommendations['batch_size'] = 16
        elif stats['total_images'] < 5000:
            recommendations['batch_size'] = 32
        else:
            recommendations['batch_size'] = 64
        
        # Embedding dimension based on number of crows
        if stats['total_crows'] < 50:
            recommendations['embedding_dim'] = 128
        elif stats['total_crows'] < 200:
            recommendations['embedding_dim'] = 256
        else:
            recommendations['embedding_dim'] = 512
        
        # Learning rate based on dataset size
        if stats['total_images'] < 2000:
            recommendations['learning_rate'] = 0.001
        else:
            recommendations['learning_rate'] = 0.0005
        
        # Epochs based on dataset complexity
        complexity_score = stats['total_crows'] * np.log(stats['images_per_crow_stats']['mean'])
        if complexity_score < 100:
            recommendations['epochs'] = 30
        elif complexity_score < 500:
            recommendations['epochs'] = 50
        else:
            recommendations['epochs'] = 100
        
        return recommendations, stats

def main():
    """Test the improved dataset."""
    # Analyze dataset
    stats = DatasetStats.analyze_dataset('crow_crops')
    recommendations, _ = DatasetStats.recommend_training_params('crow_crops')
    
    print("Dataset Statistics:")
    print(f"  Total crows: {stats['total_crows']}")
    print(f"  Total images: {stats['total_images']}")
    print(f"  Images per crow: {stats['images_per_crow_stats']}")
    
    print("\nRecommended Parameters:")
    for key, value in recommendations.items():
        print(f"  {key}: {value}")
    
    # Test dataset loading
    dataset = ImprovedCrowTripletDataset('crow_crops', split='train', transform_mode='augmented')
    print(f"\nDataset loaded: {len(dataset)} samples")
    
    # Test sample
    sample = dataset[0]
    print(f"Sample format: {type(sample)}")
    print(f"Image shapes: {[img.shape if img is not None else None for img in sample[0]]}")

if __name__ == '__main__':
    main() 