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
from sklearn.model_selection import train_test_split # For splitting crow_ids
from db import get_image_label # For DB label checking

logger = logging.getLogger(__name__)

class ImprovedCrowTripletDataset(Dataset):
    def __init__(self, base_dir, split='train', transform_mode='standard',
                 min_samples_per_crow=5, max_samples_per_crow=200,
                 curriculum_epoch=0, max_curriculum_epochs=20, model=None, class_config_path=None,
                 include_non_crow=True, non_crow_dir=None):
        """
        Improved Crow Triplet Dataset with curriculum learning and balancing, using crop_metadata.json.
        
        Args:
            base_dir: Base directory (e.g., "crow_crops") containing metadata and video crop folders.
            split: 'train' or 'val'
            transform_mode: 'standard', 'augmented', or 'heavy'
            min_samples_per_crow: Minimum samples required per crow
            max_samples_per_crow: Maximum samples to use per crow
            curriculum_epoch: Current epoch for curriculum learning
            max_curriculum_epochs: Maximum epochs for curriculum phase
            model: Optional model for hard negative mining.
            class_config_path: Optional path to class configuration JSON for balancing/grouping.
            include_non_crow: Whether to include non-crow images as negatives
            non_crow_dir: Directory for non-crow images (default: dataset/not_crow)
        """
        self.base_dir = Path(base_dir)
        self.split = split
        self.transform_mode = transform_mode
        self.min_samples_per_crow = min_samples_per_crow
        self.max_samples_per_crow = max_samples_per_crow
        self.curriculum_epoch = curriculum_epoch
        self.max_curriculum_epochs = max_curriculum_epochs
        self.model = model
        self.all_img_embeddings = {}
        self.embeddings_computed_for_model_id = None
        self.hard_negative_N_candidates = 50
        self.class_config = self._load_class_config(class_config_path) # Load class config

        self.crop_metadata_path = self.base_dir / "metadata" / "crop_metadata.json"
        self.crop_metadata = self._load_crop_metadata()

        self._setup_transforms()
        
        self.crow_to_imgs = {} # Will map crow_id to list of its selected image Paths
        self.all_image_paths_labels = [] # Tuples of (Path, crow_id) for unique images after filtering/sampling
        self.samples = [] # List of (Path, crow_id) for training, after balancing and curriculum

        self._load_and_balance_dataset()
        
        if include_non_crow:
            if non_crow_dir is None:
                non_crow_dir = Path("dataset/not_crow")
            else:
                non_crow_dir = Path(non_crow_dir)
            self._load_non_crow_samples(non_crow_dir)
        
        self._setup_curriculum() # Operates on self.samples

        if self.model is not None:
            self._ensure_embeddings_computed() # Uses self.all_image_paths_labels
        
        logger.info(f"Dataset {split}: {len(self.samples)} samples from {len(self.crow_to_imgs)} crows.")
        if not self.samples:
             logger.warning(f"Dataset {split} is empty after all processing steps.")

    def _load_crop_metadata(self):
        """Loads crop metadata from the JSON file."""
        if not self.crop_metadata_path.exists():
            logger.error(f"Crop metadata file not found: {self.crop_metadata_path}. Dataset will be empty.")
            return {"crops": {}}
        try:
            with open(self.crop_metadata_path, 'r') as f:
                metadata = json.load(f)
                logger.info(f"Successfully loaded crop metadata from {self.crop_metadata_path}")
                return metadata
        except Exception as e:
            logger.error(f"Error loading crop metadata from {self.crop_metadata_path}: {e}. Returning empty metadata.")
            return {"crops": {}}

    def _load_class_config(self, config_path):
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading class config from {config_path}: {e}")
        return None

    def _setup_transforms(self):
        """Setup data transforms based on mode."""
        # Transform for consistent embedding generation (less augmentation)
        self.embedding_transform = transforms.Compose([
            transforms.Resize((580, 580)),  # Resize to larger square to preserve aspect ratio
            transforms.CenterCrop((512, 512)),  # Crop to target size
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Base transforms for training
        base_transforms = [
            transforms.Resize((580, 580)),  # Resize to larger square to preserve aspect ratio
            transforms.CenterCrop((512, 512)),  # Crop to target size
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
        """Load and balance the dataset using crop_metadata.json and DB label filtering."""
        logger.info(f"Loading dataset from crop_metadata for {self.split} split.")

        if not self.crop_metadata or "crops" not in self.crop_metadata or not self.crop_metadata["crops"]:
            logger.warning("Crop metadata is empty or not loaded. Dataset will be empty.")
            return

        all_crow_ids_from_metadata = sorted(list(set(
            md['crow_id'] for md in self.crop_metadata["crops"].values() if md.get('crow_id')
        )))

        if not all_crow_ids_from_metadata:
            logger.warning("No crow IDs found in crop metadata. Dataset will be empty.")
            return
        
        # Perform train/val split on crow IDs
        if len(all_crow_ids_from_metadata) < 2:
            train_crow_ids = all_crow_ids_from_metadata if self.split == 'train' else []
            val_crow_ids = all_crow_ids_from_metadata if self.split == 'val' else []
            logger.warning(f"Only {len(all_crow_ids_from_metadata)} unique crow ID(s). Standard train/val split not possible.")
        else:
            try:
                train_crow_ids, val_crow_ids = train_test_split(
                    all_crow_ids_from_metadata, test_size=0.2, random_state=42 # Ensure reproducibility
                )
            except ValueError:
                 logger.warning("Train/test split failed, possibly due to too few unique crow IDs for stratification. Defaulting split logic.")
                 split_idx_fallback = int(len(all_crow_ids_from_metadata) * 0.8)
                 train_crow_ids = all_crow_ids_from_metadata[:split_idx_fallback]
                 val_crow_ids = all_crow_ids_from_metadata[split_idx_fallback:]


        selected_crow_ids_for_split = set(train_crow_ids if self.split == 'train' else val_crow_ids)
        logger.info(f"Selected {len(selected_crow_ids_for_split)} crow IDs for {self.split} split.")

        if not selected_crow_ids_for_split:
            logger.warning(f"No crow IDs selected for {self.split} split. Dataset will be empty.")
            return

        temp_crow_to_imgs_db_filtered = defaultdict(list)
        excluded_by_label_count = 0
        labels_to_exclude = {'multi_crow', 'not_a_crow', 'bad_crow'}

        logger.info(f"Processing {len(self.crop_metadata['crops'])} items from crop_metadata.")
        for crop_relative_path_str, metadata_dict in self.crop_metadata["crops"].items():
            crow_id = metadata_dict.get('crow_id')
            if not crow_id or crow_id not in selected_crow_ids_for_split:
                continue

            # Construct full_image_path using self.base_dir
            full_image_path = self.base_dir / crop_relative_path_str
            if not full_image_path.exists():
                logger.debug(f"Image path {full_image_path} from metadata does not exist. Skipping.")
                continue
            
            label_info = get_image_label(str(full_image_path)) # DB query
            if label_info and (label_info['label'] in labels_to_exclude or not label_info.get('is_training_data', True)):
                excluded_by_label_count += 1
                logger.debug(f"Excluding image {full_image_path} due to label: {label_info.get('label')} or is_training_data: {label_info.get('is_training_data')}")
                continue

            temp_crow_to_imgs_db_filtered[crow_id].append(full_image_path)

        logger.info(f"Excluded {excluded_by_label_count} images based on database labels.")

        # Apply min_samples_per_crow and max_samples_per_crow to the DB-filtered images
        self.crow_to_imgs.clear() # Ensure it's empty before populating
        for crow_id, img_paths in temp_crow_to_imgs_db_filtered.items():
            if len(img_paths) >= self.min_samples_per_crow:
                final_img_paths_for_crow = img_paths
                if self.max_samples_per_crow is not None and len(img_paths) > self.max_samples_per_crow:
                    final_img_paths_for_crow = random.sample(img_paths, self.max_samples_per_crow)
                self.crow_to_imgs[crow_id] = final_img_paths_for_crow
            else:
                logger.debug(f"Crow {crow_id} excluded from {self.split} split: has {len(img_paths)} images after DB filtering, less than min {self.min_samples_per_crow}.")

        # Populate self.all_image_paths_labels (unique images for embedding) and self.samples (for training iteration)
        self.all_image_paths_labels.clear()
        self.samples.clear()
        for crow_id, img_paths_list in self.crow_to_imgs.items():
            for img_path in img_paths_list:
                self.all_image_paths_labels.append((img_path, crow_id))
                self.samples.append((img_path, crow_id)) # Initial samples before class balancing

        if not self.samples:
            logger.warning(f"No samples available for {self.split} split after all filtering and sampling based on crop_metadata. Dataset will be empty.")
            # No need to call _balance_classes if no samples.
            return

        logger.info(f"Initially loaded {len(self.samples)} samples from {len(self.crow_to_imgs)} crows for {self.split} split using crop_metadata.")

        # Call balancing methods.
        # The original file had one _balance_classes for oversampling.
        # The prompt suggests a _balance_classes that uses class_config.
        # I will assume that if class_config exists, it dictates remapping and specific balancing.
        # Otherwise, the old oversampling logic is used.
        if self.class_config and 'classes' in self.class_config:
             self._balance_classes_with_config()
        else:
             self._balance_classes_by_oversampling()

        logger.info(f"After processing and balancing, using {len(self.samples)} samples from {len(self.crow_to_imgs)} effective classes for {self.split} split.")

        # Log class distribution of final self.samples
        final_class_counts = defaultdict(int)
        for _, crow_id_label in self.samples: # crow_id_label could be original or remapped
            final_class_counts[crow_id_label] += 1

        if final_class_counts:
            logger.info(f"Final class distribution for {self.split} split: "
                       f"min samples/class={min(final_class_counts.values()) if final_class_counts else 0}, "
                       f"max samples/class={max(final_class_counts.values()) if final_class_counts else 0}, "
                       f"mean samples/class={np.mean(list(final_class_counts.values())) if final_class_counts else 0:.1f}, "
                       f"total classes={len(final_class_counts)}")
        else:
            logger.info(f"Final class distribution for {self.split} split: No samples found.")

    def _balance_classes_by_oversampling(self): # Renamed original _balance_classes (was previously just _balance_classes)
        """Balance classes by oversampling based on 75th percentile of current sample counts."""
        logger.info("Balancing classes by oversampling (75th percentile)...")
        
        if not self.samples:
            logger.warning("No samples available to balance by oversampling.")
            return

        # self.crow_to_imgs contains the available images for each class (original crow_id or remapped class name)
        # self.samples contains the current list of (image_path, class_label) tuples to be balanced.
        
        current_sample_counts_per_class = defaultdict(int)
        for _, class_label in self.samples:
            current_sample_counts_per_class[class_label] += 1
        
        if not current_sample_counts_per_class:
            logger.warning("No class counts from current samples. Skipping oversampling.")
            return

        counts_values = list(current_sample_counts_per_class.values())
        target_count = int(np.percentile(counts_values, 75))
        logger.info(f"Target count for oversampling: {target_count} (75th percentile of current sample counts per class).")

        balanced_samples_list = []
        for class_label, available_imgs_for_class in self.crow_to_imgs.items():
            # available_imgs_for_class are the unique images for this class after initial sampling (min/max per crow)
            if not available_imgs_for_class:
                logger.debug(f"No images available in self.crow_to_imgs for class {class_label} to use for oversampling.")
                continue

            # Add existing samples for this class first (those that are in self.samples)
            # This ensures that if a class already meets/exceeds target, its current samples are preserved.
            existing_samples_for_this_class = [s for s in self.samples if s[1] == class_label]
            balanced_samples_list.extend(existing_samples_for_this_class)

            num_to_add = 0
            current_count_for_this_class = len(existing_samples_for_this_class) # current_sample_counts_per_class.get(class_label,0)

            if current_count_for_this_class < target_count:
                num_to_add = target_count - current_count_for_this_class

            if num_to_add > 0:
                additional_samples_needed = num_to_add
                # Draw additional samples from available_imgs_for_class
                while additional_samples_needed > 0:
                    take = min(additional_samples_needed, len(available_imgs_for_class))
                    balanced_samples_list.extend([(img_p, class_label) for img_p in random.sample(available_imgs_for_class, take)])
                    additional_samples_needed -= take

        self.samples = balanced_samples_list
        logger.info(f"After oversampling, dataset has {len(self.samples)} samples.")

    def _balance_classes_with_config(self):
        """Balances classes based on the provided class configuration (remapping and then min/max sampling)."""
        logger.info("Balancing classes using provided class configuration...")
        if not self.class_config or 'classes' not in self.class_config:
            logger.warning("Class config not provided or malformed for _balance_classes_with_config. Defaulting to simple oversampling if any samples exist.")
            if self.samples: # If there are samples from initial load
                 self._balance_classes_by_oversampling()
            return

        # self.crow_to_imgs currently holds original crow_ids and their filtered/sampled images
        # self.samples also holds (path, original_crow_id)

        new_crow_to_imgs_remapped = defaultdict(list)
        
        # Create mappings for new balanced classes from config
        # These will become the new class labels if config is used.
        self.class_to_idx = {cfg["name"]: i for i, cfg in enumerate(self.class_config["classes"])}
        self.idx_to_class = {i: cfg["name"] for i, cfg in enumerate(self.class_config["classes"])}

        # Assign images to new classes based on original crow_id
        for original_crow_id, imgs_list_for_original_id in self.crow_to_imgs.items():
            matched_to_new_class = False
            for new_class_cfg in self.class_config["classes"]:
                if original_crow_id in new_class_cfg.get("original_ids", []):
                    new_crow_to_imgs_remapped[new_class_cfg["name"]].extend(imgs_list_for_original_id)
                    matched_to_new_class = True
            if not matched_to_new_class and self.class_config.get("include_unlisted_classes", False):
                logger.debug(f"Original crow ID {original_crow_id} not mapped to any new class. Images dropped as per current include_unlisted_classes handling.")

        # Apply min/max sampling for each new remapped class
        final_remapped_crow_to_imgs = {}
        new_samples_list = []

        for new_class_name, all_imgs_for_new_class_list in new_crow_to_imgs_remapped.items():
            unique_imgs_for_new_class = sorted(list(set(all_imgs_for_new_class_list)), key=lambda p: str(p))

            class_cfg_list = [c for c in self.class_config["classes"] if c["name"] == new_class_name]
            if not class_cfg_list: continue
            class_cfg = class_cfg_list[0]

            min_s = class_cfg.get("min_samples", self.min_samples_per_crow) # Use class-specific or dataset global
            max_s = class_cfg.get("max_samples", self.max_samples_per_crow) # Use class-specific or dataset global

            if len(unique_imgs_for_new_class) >= min_s:
                sampled_imgs_for_new_class = unique_imgs_for_new_class
                if max_s is not None and len(unique_imgs_for_new_class) > max_s:
                    sampled_imgs_for_new_class = random.sample(unique_imgs_for_new_class, max_s)

                final_remapped_crow_to_imgs[new_class_name] = sampled_imgs_for_new_class
                for img_path in sampled_imgs_for_new_class:
                    new_samples_list.append((img_path, new_class_name)) # Label is new_class_name
            else:
                logger.debug(f"New class {new_class_name} from config excluded/emptied: has {len(unique_imgs_for_new_class)} unique images, less than min {min_s}.")

        self.crow_to_imgs = final_remapped_crow_to_imgs # self.crow_to_imgs now uses new class names as keys
        self.samples = new_samples_list # self.samples now uses new class names as labels

        # Update self.all_image_paths_labels to reflect remapped class names and final images
        self.all_image_paths_labels.clear()
        for class_label, img_paths in self.crow_to_imgs.items():
            for img_path in img_paths:
                self.all_image_paths_labels.append((img_path, class_label))

        if not self.samples:
            logger.warning(f"Dataset became empty after class config balancing for {self.split} split.")
        else:
            logger.info(f"After class config balancing, {len(self.samples)} samples in {len(self.crow_to_imgs)} remapped classes for {self.split} split.")
    
    def _balance_classes(self): # This is the original method name, now a router
        """Balance classes either by config or by oversampling."""
        if self.class_config and 'classes' in self.class_config:
            self._balance_classes_with_config()
        elif self.samples: # Only oversample if there are samples to begin with
            self._balance_classes_by_oversampling()
        else:
            logger.info("No class_config and no initial samples, skipping balancing.")
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

    def _load_non_crow_samples(self, non_crow_dir):
        """Load non-crow images as negative examples"""
        if not non_crow_dir.exists():
            logger.warning(f"Non-crow directory not found: {non_crow_dir}")
            return
        non_crow_images = []
        for img_path in non_crow_dir.glob("*.jpg"):
            non_crow_images.append((img_path, "not_a_crow"))
        for img_path in non_crow_dir.glob("*.jpeg"):
            non_crow_images.append((img_path, "not_a_crow"))
        for img_path in non_crow_dir.glob("*.png"):
            non_crow_images.append((img_path, "not_a_crow"))
        self.samples.extend(non_crow_images)
        logger.info(f"Added {len(non_crow_images)} non-crow images as negative examples from {non_crow_dir}")

class DatasetStats:
    """Utility class for dataset statistics."""
    
    @staticmethod
    def analyze_dataset(base_dir): # Changed crop_dir to base_dir
        """Analyze dataset using crop_metadata.json and DB label filtering, and provide statistics."""
        base_path = Path(base_dir)
        metadata_path = base_path / "metadata" / "crop_metadata.json"

        if not metadata_path.exists():
            logger.error(f"Metadata file not found for analysis: {metadata_path}. Returning empty stats.")
            return {
                'total_crows': 0, 'total_images': 0, 'images_per_crow': [],
                'image_sizes': [], 'file_sizes': [],
                'images_per_crow_stats': {}, 'file_size_stats': {}
            }

        try:
            with open(metadata_path, 'r') as f:
                crop_metadata = json.load(f)
        except Exception as e:
            logger.error(f"Error loading metadata file {metadata_path}: {e}. Returning empty stats.")
            return {
                'total_crows': 0, 'total_images': 0, 'images_per_crow': [],
                'image_sizes': [], 'file_sizes': [],
                'images_per_crow_stats': {}, 'file_size_stats': {}
            }

        stats = {
            'total_crows': 0, 'total_images': 0, 'images_per_crow': [],
            'image_sizes': [], 'file_sizes': [], 'excluded_by_label': 0
        }
        
        crow_to_valid_images = defaultdict(list)
        labels_to_exclude = {'multi_crow', 'not_a_crow', 'bad_crow'}

        if "crops" not in crop_metadata:
            logger.warning(f"No 'crops' key in metadata file: {metadata_path}. Returning empty stats.")
            return stats # Already initialized with zeros

        for crop_relative_path_str, metadata_dict in crop_metadata["crops"].items():
            full_image_path = base_path / crop_relative_path_str
            
            if not full_image_path.exists():
                logger.debug(f"Image path {full_image_path} from metadata does not exist during analysis. Skipping.")
                continue

            label_info = get_image_label(str(full_image_path)) # DB query
            if label_info and (label_info['label'] in labels_to_exclude or not label_info.get('is_training_data', True)):
                stats['excluded_by_label'] += 1
                continue

            crow_id = metadata_dict.get('crow_id')
            if crow_id: # Ensure crow_id exists
                crow_to_valid_images[crow_id].append(full_image_path)

        stats['total_crows'] = len(crow_to_valid_images)
        for crow_id, img_list in crow_to_valid_images.items():
            if not img_list: continue # Should not happen if crow_id is a key

            stats['total_images'] += len(img_list)
            stats['images_per_crow'].append(len(img_list))

            # Sample up to 10 images from this crow's valid list for size analysis
            sample_size = min(len(img_list), 10)
            sampled_images = random.sample(img_list, sample_size)

            for img_file_path in sampled_images:
                try:
                    # Ensure img_file_path is a Path object if not already
                    img_p = Path(img_file_path)
                    img = Image.open(img_p)
                    stats['image_sizes'].append(img.size)
                    stats['file_sizes'].append(img_p.stat().st_size)
                except FileNotFoundError:
                    logger.warning(f"File not found during stat analysis: {img_p}")
                except Exception as e: # Catch other PIL or stat errors
                    logger.warning(f"Could not analyze file {img_p}: {e}")
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
    def recommend_training_params(base_dir): # Changed crop_dir to base_dir
        """Recommend training parameters based on dataset analysis."""
        stats = DatasetStats.analyze_dataset(base_dir) # Pass base_dir
        
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
    # Analyze dataset using base_dir
    stats = DatasetStats.analyze_dataset('crow_crops') # Assuming 'crow_crops' is the base_dir
    if stats['total_images'] > 0 : # Only print if stats are meaningful
        recommendations, _ = DatasetStats.recommend_training_params('crow_crops') # Pass base_dir

        print("Dataset Statistics (from crop_metadata and DB filtering):")
        print(f"  Total valid crows: {stats['total_crows']}")
        print(f"  Total valid images: {stats['total_images']}")
        print(f"  Images excluded by DB label: {stats.get('excluded_by_label', 0)}")
        if stats['images_per_crow_stats']:
             print(f"  Images per crow: {stats['images_per_crow_stats']}")

        print("\nRecommended Parameters:")
        for key, value in recommendations.items():
            print(f"  {key}: {value}")
    else:
        print("Dataset analysis yielded no images. Cannot provide recommendations or detailed stats.")
    
    # Test dataset loading (now expects base_dir)
    # Ensure a class_config_path is provided if your test setup needs it, or None
    test_dataset = ImprovedCrowTripletDataset(base_dir='crow_crops', split='train',
                                            transform_mode='augmented', class_config_path=None)
    print(f"\nTest Dataset loaded: {len(test_dataset)} samples")
    
    if len(test_dataset) > 0:
        sample = test_dataset[0]
        print(f"Sample format: {type(sample)}")
        # Assuming sample[0] are the images (anchor, positive, negative)
        if isinstance(sample[0], tuple) and len(sample[0]) == 3:
             print(f"Image shapes: {[img.shape if hasattr(img, 'shape') else type(img) for img in sample[0]]}")
        else:
             print(f"Sample images format not as expected: {type(sample[0])}")
    else:
        print("Test dataset is empty, cannot retrieve a sample.")


if __name__ == '__main__':
    main() 