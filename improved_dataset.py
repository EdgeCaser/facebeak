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
                 curriculum_epoch=0, max_curriculum_epochs=20, model=None): # Added model parameter
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
        self.model = model # Store the model
        self.all_img_embeddings = {} # To store pre-computed embeddings {path: tensor}
        self.embeddings_computed_for_model_id = None # id(model) for which embeddings were computed
        self.hard_negative_N_candidates = 50 # Number of candidates for hard negative mining

        # Setup transforms (including a specific one for embedding generation)
        self._setup_transforms()
        
        # Load and balance dataset
        self._load_and_balance_dataset() # This defines self.crow_to_imgs, 
                                         # self.all_image_paths_labels, and self.samples (balanced)
        
        # Setup curriculum learning (operates on self.samples)
        self._setup_curriculum()

        # Pre-compute embeddings if model is provided
        if self.model is not None:
            self._ensure_embeddings_computed()
        
        logger.info(f"Dataset {split}: {len(self.samples)} samples from {len(self.crow_to_imgs)} crows")
        
    def _setup_transforms(self):
        """Setup data transforms based on mode."""
        # Transform for consistent embedding generation (less augmentation)
        self.embedding_transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Base transforms for training
        base_transforms = [
            transforms.Resize((512, 512)), 
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
        
        if self.transform_mode == 'standard':
            self.transform = transforms.Compose(base_transforms)
            
        elif self.transform_mode == 'augmented':
            augmented_transforms = [
                transforms.Resize((580, 580)),
                transforms.RandomCrop((512, 512)),
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
                transforms.Resize((580, 580)),
                transforms.RandomCrop((512, 512)),
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
            img_files = list(crow_dir.glob("*.jpg")) + list(crow_dir.glob("*.png")) + list(crow_dir.glob("*.jpeg"))
            
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

        # Create a list of all unique image paths and their labels from the selected crows/images
        # This list is used for computing embeddings.
        self.all_image_paths_labels = []
        for crow_id, img_files_list in self.crow_to_imgs.items():
            for img_path in img_files_list:
                self.all_image_paths_labels.append((img_path, crow_id))
        
        # Create initial sample list for training (anchor selection pool)
        # This list will be balanced. self.crow_to_imgs here contains images up to max_samples_per_crow.
        self.samples = [] 
        for crow_id, img_files_list in self.crow_to_imgs.items(): 
            for img_path in img_files_list:
                self.samples.append((img_path, crow_id))

        self._balance_classes() # Balances self.samples by oversampling based on target counts
        
        logger.info(f"Loaded {len(self.all_image_paths_labels)} unique images from {len(self.crow_to_imgs)} crows. "
                    f"Balanced training samples: {len(self.samples)}.")
        
        # Log class distribution
        class_counts = defaultdict(int)
        for _, crow_id in self.samples:
            class_counts[crow_id] += 1
        
        if class_counts:
            logger.info(f"Class distribution: min={min(class_counts.values())}, "
                       f"max={max(class_counts.values())}, "
                       f"mean={np.mean(list(class_counts.values())):.1f}")
        else:
            logger.info("Class distribution: No samples found")
    
    def _balance_classes(self):
        """Balance classes by oversampling."""
        # Count samples per crow
        crow_counts = defaultdict(int)
        for _, crow_id in self.samples:
            crow_counts[crow_id] += 1
        
        # Find target count (75th percentile)
        counts = list(crow_counts.values())
        if not counts:
            logger.warning("No samples found for balancing. Skipping class balancing.")
            return
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

    def set_model(self, model):
        """Sets the model and flags that embeddings need recomputation."""
        if self.model is not model: # Check if it's actually a new model instance
            logger.info("Model updated in dataset. Embeddings will be recomputed if accessed for hard negative mining.")
            self.model = model
            # Mark embeddings as stale by mismatching the ID or clearing them
            self.embeddings_computed_for_model_id = None 
            # self.all_img_embeddings.clear() # Optionally clear, or let _ensure_embeddings_computed handle it

    def _ensure_embeddings_computed(self):
        """
        Ensures that embeddings are computed for all unique images using the current model.
        Returns True if embeddings are ready, False otherwise.
        """
        if self.model is None:
            logger.debug("No model set in dataset. Cannot compute embeddings for hard negative mining.")
            return False
        
        current_model_id = id(self.model)
        if self.embeddings_computed_for_model_id == current_model_id and self.all_img_embeddings:
            # Embeddings are current and available.
            return True 

        logger.info(f"Computing/Re-computing embeddings for hard negative mining with model ID {current_model_id}...")
        new_embeddings_cache = {}
        
        original_model_training_state = self.model.training
        self.model.eval() # Set model to evaluation mode
        
        try:
            device = next(self.model.parameters()).device # Get model's device
        except StopIteration: # Model has no parameters
            logger.error("Model has no parameters. Cannot compute embeddings.")
            if original_model_training_state: self.model.train() # Restore state
            return False


        with torch.no_grad():
            for img_path, _ in self.all_image_paths_labels: # Iterate over unique images
                try:
                    img = Image.open(img_path).convert('RGB')
                    img_tensor = self.embedding_transform(img).unsqueeze(0) 
                    img_tensor = img_tensor.to(device)
                    
                    embedding = self.model(img_tensor)
                    new_embeddings_cache[img_path] = embedding.squeeze(0).cpu() # Store on CPU
                except Exception as e:
                    logger.error(f"Failed to compute embedding for {img_path}: {e}")
        
        self.all_img_embeddings = new_embeddings_cache
        self.embeddings_computed_for_model_id = current_model_id
        
        if original_model_training_state: # Restore model's original training state
            self.model.train()
            
        logger.info(f"Computed and cached {len(self.all_img_embeddings)} embeddings.")
        return bool(self.all_img_embeddings) # Return true if some embeddings were computed

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
            return torch.randn(3, 512, 512)
    
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
        
        embeddings_ready = self._ensure_embeddings_computed()

        if not embeddings_ready or anchor_embedding is None or not self.all_img_embeddings:
            if self.model is not None and (not embeddings_ready or not self.all_img_embeddings):
                 logger.debug(f"HNM for {anchor_crow_id}: Falling back to random negative (embeddings_ready={embeddings_ready}, anchor_emb_is_none={anchor_embedding is None}, cache_empty={not self.all_img_embeddings}).")
            return self._get_negative_sample(anchor_crow_id)

        hard_negative_path = None
        min_dist = float('inf')
        
        anchor_embedding_cpu = anchor_embedding.cpu() # Ensure anchor embedding is on CPU for distance calc

        # Collect all valid negative candidate paths that have embeddings
        negative_candidates_paths = []
        for path, label in self.all_image_paths_labels: # Iterate unique images
            if label != anchor_crow_id and path in self.all_img_embeddings:
                negative_candidates_paths.append(path)
        
        if not negative_candidates_paths:
            logger.debug(f"HNM for {anchor_crow_id}: No valid negative candidates with embeddings found. Falling back.")
            return self._get_negative_sample(anchor_crow_id)

        # If more candidates than desired, sample a subset
        if len(negative_candidates_paths) > self.hard_negative_N_candidates:
            selected_candidate_paths = random.sample(negative_candidates_paths, self.hard_negative_N_candidates)
        else:
            selected_candidate_paths = negative_candidates_paths
            
        for neg_path in selected_candidate_paths:
            neg_embedding = self.all_img_embeddings.get(neg_path) # Should exist due to earlier check
            # neg_embedding is already on CPU as stored in cache
            dist = torch.norm(anchor_embedding_cpu - neg_embedding, p=2).item() # L2 distance
            if dist < min_dist:
                min_dist = dist
                hard_negative_path = neg_path
        
        if hard_negative_path:
            # logger.debug(f"HNM for {anchor_crow_id}: Selected {hard_negative_path} (dist: {min_dist:.4f})")
            return hard_negative_path
        else:
            # Fallback if no suitable hard negative found in candidates
            logger.debug(f"HNM for {anchor_crow_id}: No hard negative found from candidates, falling back to random.")
            return self._get_negative_sample(anchor_crow_id)

    def __getitem__(self, idx):
        """Get a triplet sample."""
        anchor_path, anchor_crow_id = self.samples[idx]
        anchor_img = self._load_image(anchor_path) # This applies self.transform (training augmentations)

        anchor_embedding_for_hnm = None
        if self.model: 
            try:
                original_model_training_state = self.model.training
                self.model.eval()
                with torch.no_grad():
                    # Use embedding_transform for consistency with pre-computed embeddings
                    raw_anchor_img_pil = Image.open(anchor_path).convert('RGB')
                    anchor_emb_tensor_for_hnm_device = self.embedding_transform(raw_anchor_img_pil).unsqueeze(0)
                    
                    device = next(self.model.parameters()).device # Get model's device
                    anchor_emb_tensor_for_hnm_device = anchor_emb_tensor_for_hnm_device.to(device)
                    
                    # The embedding passed to _get_hard_negative_sample should be on the model's device initially,
                    # as it will be .cpu()'d within that method.
                    anchor_embedding_for_hnm = self.model(anchor_emb_tensor_for_hnm_device).squeeze(0) 
                if original_model_training_state: # Restore state
                    self.model.train()
            except Exception as e:
                logger.error(f"Could not compute anchor embedding for HNM for {anchor_path}: {e}")
        
        pos_path = self._get_positive_sample(anchor_crow_id, anchor_path)
        pos_img = self._load_image(pos_path)
        
        # Determine whether to use hard negative mining.
        # Could be based on curriculum_epoch, e.g., enable after some initial epochs.
        # For now, always attempt if model is available.
        use_hard_negatives = self.model is not None
        
        if use_hard_negatives:
            neg_path = self._get_hard_negative_sample(anchor_crow_id, anchor_embedding=anchor_embedding_for_hnm)
        else:
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
            img_files = list(crow_dir.glob("*.jpg")) + list(crow_dir.glob("*.png")) + list(crow_dir.glob("*.jpeg"))
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