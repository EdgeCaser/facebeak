import os
import torch
from torch.utils.data import DataLoader, Dataset, random_split
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import cv2
import numpy as np
from models import CrowResNetEmbedder
from torchvision import transforms
import logging
from pathlib import Path
import json
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import random

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class CrowTripletDataset(Dataset):
    def __init__(self, crop_dir, transform=None, split='train'):
        """
        Dataset for training crow embeddings using triplet loss.
        Args:
            crop_dir: Directory containing crow images. Can be:
                     - A directory with crow ID subdirectories
                     - A directory containing multiple session folders
                     - A directory containing images directly (will be treated as one crow)
            transform: Optional torchvision transforms
            split: 'train' or 'val' to determine which transforms to use
        """
        self.samples = []
        self.crow_to_imgs = {}
        self.split = split
        logger.debug(f"Initializing dataset from {crop_dir} for {split} split")
        
        # Define transforms based on split
        if transform is None:
            if split == 'train':
                self.transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(30),
                    transforms.RandomAffine(
                        degrees=0, 
                        translate=(0.1, 0.1),
                        scale=(0.9, 1.1)
                    ),
                    transforms.ColorJitter(
                        brightness=0.3,
                        contrast=0.3,
                        saturation=0.3,
                        hue=0.1
                    ),
                    transforms.RandomErasing(p=0.3),
                    transforms.RandomApply([
                        transforms.GaussianBlur(kernel_size=3)
                    ], p=0.2),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]
                    )
                ])
                logger.debug("Using training transforms with augmentation")
            else:  # val/test
                self.transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]
                    )
                ])
                logger.debug("Using validation transforms (no augmentation)")
        else:
            self.transform = transform
            logger.debug("Using custom transforms")
        
        # Load all crow images recursively
        def process_directory(current_dir, session_name="", crow_id=None):
            """Recursively process directories to find crow images."""
            logger.debug(f"Processing directory: {current_dir} (session: {session_name}, crow: {crow_id})")
            
            # Check if this directory contains images directly
            has_images = any(f.endswith(('.jpg', '.png')) for f in os.listdir(current_dir))
            has_subdirs = any(os.path.isdir(os.path.join(current_dir, d)) for d in os.listdir(current_dir))
            
            if has_images and not has_subdirs:
                # This is a directory with images directly
                if crow_id is None:
                    # Use directory name as crow ID
                    crow_id = os.path.basename(current_dir)
                process_crow_directory(current_dir, crow_id, session_name)
            else:
                # Process subdirectories
                for item in os.listdir(current_dir):
                    item_path = os.path.join(current_dir, item)
                    
                    if os.path.isdir(item_path):
                        # If this is a session directory (contains crow subdirectories)
                        if any(os.path.isdir(os.path.join(item_path, d)) for d in os.listdir(item_path)):
                            logger.debug(f"Found session directory: {item}")
                            process_directory(item_path, item)
                        else:
                            # This might be a crow directory
                            process_directory(item_path, session_name, item)
                    elif item.endswith(('.jpg', '.png')):
                        # Found an image in a directory
                        if crow_id is None:
                            # Use directory name as crow ID
                            crow_id = os.path.basename(current_dir)
                        logger.debug(f"Found image {item} in directory {current_dir}, using crow_id: {crow_id}")
                        process_crow_directory(current_dir, crow_id, session_name)
                        break  # Process all images in this directory
        
        def process_crow_directory(crow_dir, crow_id, session_name=""):
            """Process a directory containing crow images."""
            if not os.path.isdir(crow_dir):
                return
                
            logger.debug(f"Processing crow directory: {crow_dir} (crow_id: {crow_id}, session: {session_name})")
            imgs = []
            for img_name in os.listdir(crow_dir):
                if not img_name.endswith(('.jpg', '.png')):
                    continue
                    
                img_path = os.path.join(crow_dir, img_name)
                try:
                    # Load and preprocess image
                    img = cv2.imread(img_path)
                    if img is None:
                        logger.warning(f"Failed to load image: {img_path}")
                        continue
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    # Resize maintaining aspect ratio
                    h, w = img.shape[:2]
                    target_size = 224
                    scale = target_size / max(h, w)
                    new_h, new_w = int(h * scale), int(w * scale)
                    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                    
                    # Pad to square
                    square_img = np.zeros((target_size, target_size, 3), dtype=np.uint8)
                    y_offset = (target_size - new_h) // 2
                    x_offset = (target_size - new_w) // 2
                    square_img[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = img
                    
                    imgs.append(square_img)
                    logger.debug(f"Successfully processed image: {img_path}")
                except Exception as e:
                    logger.error(f"Error processing {img_path}: {e}", exc_info=True)
                    continue
            
            if len(imgs) >= 2:  # Need at least 2 images for triplet
                # Use session name in crow ID if available
                full_crow_id = f"{session_name}_{crow_id}" if session_name else crow_id
                if full_crow_id not in self.crow_to_imgs:
                    self.crow_to_imgs[full_crow_id] = []
                self.crow_to_imgs[full_crow_id].extend(imgs)
                for img in imgs:
                    self.samples.append((full_crow_id, img))
                logger.info(f"Added {len(imgs)} images for crow {full_crow_id}")
            else:
                logger.warning(f"Skipping crow {crow_id} - insufficient images ({len(imgs)})")
        
        # Start recursive processing
        process_directory(crop_dir)
        
        self.crow_ids = list(self.crow_to_imgs.keys())
        logger.info(f"Loaded {len(self.samples)} images from {len(self.crow_ids)} crows for {split} split")
        if len(self.crow_ids) > 0:
            logger.info(f"Crow IDs: {', '.join(self.crow_ids)}")
            for crow_id, imgs in self.crow_to_imgs.items():
                logger.info(f"Crow {crow_id}: {len(imgs)} images")
                if len(imgs) < 2:
                    logger.warning(f"Crow {crow_id} has fewer than 2 images and will be skipped for triplet training.")
        else:
            logger.error("No valid crow images found in the dataset!")
        # --- NEW: dataset validation ---
        if len(self.crow_ids) < 2:
            logger.error("Triplet training requires at least 2 different crow IDs (classes) in your dataset.")
            raise ValueError("Triplet training requires at least 2 different crow IDs (classes) in your dataset.")
        for crow_id, imgs in self.crow_to_imgs.items():
            if len(imgs) < 2:
                logger.error(f"Crow ID '{crow_id}' has fewer than 2 images. Each crow must have at least 2 images for triplet training.")
                raise ValueError(f"Crow ID '{crow_id}' has fewer than 2 images. Each crow must have at least 2 images for triplet training.")
        logger.info(f"Final valid crows: {len(self.crow_ids)}")
        logger.info(f"Final total images: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        anchor_crow, anchor_img = self.samples[idx]
        
        # Get positive sample (different image of same crow)
        pos_img = anchor_img
        crow_imgs = self.crow_to_imgs[anchor_crow]
        if len(crow_imgs) < 2:
            raise ValueError(f"Crow ID '{anchor_crow}' has fewer than 2 images. Cannot sample positive.")
        while True:
            pos_img_candidate = random.choice(crow_imgs)
            if not np.array_equal(pos_img_candidate, anchor_img):
                pos_img = pos_img_candidate
                break
        
        # Get negative sample (image of different crow)
        neg_crow = anchor_crow
        while neg_crow == anchor_crow:
            neg_crow = random.choice(self.crow_ids)
        neg_img = random.choice(self.crow_to_imgs[neg_crow])
        
        # Apply transforms
        if self.transform:
            anchor_img = self.transform(anchor_img)
            pos_img = self.transform(pos_img)
            neg_img = self.transform(neg_img)
        
        return anchor_img, pos_img, neg_img, anchor_crow

class TripletLoss(nn.Module):
    def __init__(self, margin=1.0, mining_type='hard'):
        super().__init__()
        self.margin = margin
        self.mining_type = mining_type
        self.loss_fn = nn.TripletMarginLoss(margin=margin, p=2)
    
    def forward(self, anchor, positive, negative):
        if self.mining_type == 'hard':
            # Calculate all pairwise distances
            pos_dist = torch.norm(anchor - positive, p=2, dim=1)
            neg_dist = torch.norm(anchor - negative, p=2, dim=1)
            # Find hardest negative (closest negative that still violates margin)
            hardest_negative_idx = torch.argmin(neg_dist)
            # Keep batch dimension for hardest negative
            hardest_negative = negative[hardest_negative_idx:hardest_negative_idx+1]
            # Use hardest negative for loss
            return self.loss_fn(anchor, positive, hardest_negative)
        else:
            # Use all triplets
            return self.loss_fn(anchor, positive, negative)

def compute_metrics(model, val_loader, device):
    """Compute validation metrics."""
    model.eval()
    all_embeddings = []
    all_labels = []
    
    with torch.no_grad():
        for anchor, pos, neg, label in val_loader:
            anchor = anchor.to(device)
            pos = pos.to(device)
            neg = neg.to(device)
            anchor_emb = model(anchor).cpu().numpy()
            pos_emb = model(pos).cpu().numpy()
            neg_emb = model(neg).cpu().numpy()
            # label is a list of crow IDs (strings), one per sample in batch
            # For each batch, label is a list of strings of length batch_size
            # We need to extend all_labels with the batch's labels, repeated for anchor, pos, neg
            all_embeddings.extend([anchor_emb, pos_emb, neg_emb])
            # label is a list of strings, so we need to flatten and repeat for anchor, pos, neg
            if isinstance(label, (list, tuple)):
                all_labels.extend(label)
                all_labels.extend(label)
                all_labels.extend(label)
            else:
                # fallback: single label
                all_labels.extend([label, label, label])
    # Convert to numpy arrays and ensure proper shapes
    all_embeddings = np.vstack(all_embeddings)  # (N, embedding_dim)
    # Map string labels to unique integers for similarity computation
    unique_labels = list(sorted(set(all_labels)))
    label_to_int = {lbl: idx for idx, lbl in enumerate(unique_labels)}
    all_labels_int = np.array([label_to_int[lbl] for lbl in all_labels])
    if len(all_embeddings) == 0 or len(all_labels_int) == 0:
        logger.warning("No embeddings or labels found in validation data")
        return {
            'mean_similarity': 0.0,
            'std_similarity': 0.0,
            'min_similarity': 0.0,
            'max_similarity': 0.0,
            'same_crow_mean': 0.0,
            'same_crow_std': 0.0,
            'diff_crow_mean': 0.0,
            'diff_crow_std': 0.0
        }, np.array([]), np.array([])
    similarities = cosine_similarity(all_embeddings)
    same_crow_mask = (all_labels_int[:, None] == all_labels_int[None, :]) & ~np.eye(len(all_labels_int), dtype=bool)
    diff_crow_mask = (all_labels_int[:, None] != all_labels_int[None, :])
    same_crow_sims = similarities[same_crow_mask]
    diff_crow_sims = similarities[diff_crow_mask]
    metrics = {
        'mean_similarity': float(np.mean(similarities)),
        'std_similarity': float(np.std(similarities)),
        'min_similarity': float(np.min(similarities)),
        'max_similarity': float(np.max(similarities)),
        'same_crow_mean': float(np.mean(same_crow_sims)) if len(same_crow_sims) > 0 else 0.0,
        'same_crow_std': float(np.std(same_crow_sims)) if len(same_crow_sims) > 0 else 0.0,
        'diff_crow_mean': float(np.mean(diff_crow_sims)) if len(diff_crow_sims) > 0 else 0.0,
        'diff_crow_std': float(np.std(diff_crow_sims)) if len(diff_crow_sims) > 0 else 0.0
    }
    return metrics, similarities, all_labels_int

def plot_metrics(metrics_history, output_dir):
    """Plot training metrics."""
    plt.figure(figsize=(15, 10))
    
    # Plot loss
    plt.subplot(2, 2, 1)
    plt.plot(metrics_history['train_loss'], label='Train')
    plt.plot(metrics_history['val_loss'], label='Validation')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot similarities
    plt.subplot(2, 2, 2)
    plt.plot(metrics_history['same_crow_mean'], label='Same Crow')
    plt.plot(metrics_history['diff_crow_mean'], label='Different Crow')
    plt.fill_between(
        range(len(metrics_history['same_crow_mean'])),
        np.array(metrics_history['same_crow_mean']) - np.array(metrics_history['same_crow_std']),
        np.array(metrics_history['same_crow_mean']) + np.array(metrics_history['same_crow_std']),
        alpha=0.2
    )
    plt.fill_between(
        range(len(metrics_history['diff_crow_mean'])),
        np.array(metrics_history['diff_crow_mean']) - np.array(metrics_history['diff_crow_std']),
        np.array(metrics_history['diff_crow_mean']) + np.array(metrics_history['diff_crow_std']),
        alpha=0.2
    )
    plt.title('Embedding Similarities')
    plt.xlabel('Epoch')
    plt.ylabel('Cosine Similarity')
    plt.legend()
    
    # Plot similarity distributions
    plt.subplot(2, 2, 3)
    sns.kdeplot(metrics_history['val_similarities'][-1], label='Validation')
    plt.title('Similarity Distribution (Last Epoch)')
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Density')
    
    # Save plot
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_metrics.png'))
    plt.close()

def train_model(model, train_loader, val_loader, criterion, optimizer, device, 
                epochs, output_dir, patience=5):
    """Train the model using triplet loss with validation and early stopping."""
    best_val_loss = float('inf')
    patience_counter = 0
    metrics_history = {
        'train_loss': [],
        'val_loss': [],
        'same_crow_mean': [],
        'same_crow_std': [],
        'diff_crow_mean': [],
        'diff_crow_std': [],
        'val_similarities': []
    }
    
    for epoch in range(epochs):
        # Training
        model.train()
        total_loss = 0
        for anchor, positive, negative, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            # Move to device
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            anchor_emb = model(anchor)
            pos_emb = model(positive)
            neg_emb = model(negative)
            
            # Compute triplet loss
            loss = criterion(anchor_emb, pos_emb, neg_emb)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * anchor.size(0)
        
        avg_train_loss = total_loss / len(train_loader.dataset)
        metrics_history['train_loss'].append(avg_train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for anchor, positive, negative, _ in val_loader:
                anchor = anchor.to(device)
                positive = positive.to(device)
                negative = negative.to(device)
                
                anchor_emb = model(anchor)
                pos_emb = model(positive)
                neg_emb = model(negative)
                
                loss = criterion(anchor_emb, pos_emb, neg_emb)
                val_loss += loss.item() * anchor.size(0)
        
        avg_val_loss = val_loss / len(val_loader.dataset)
        metrics_history['val_loss'].append(avg_val_loss)
        
        # Compute validation metrics
        val_metrics, similarities, _ = compute_metrics(model, val_loader, device)
        metrics_history['val_similarities'].append(similarities)
        metrics_history['same_crow_mean'].append(val_metrics['same_crow_mean'])
        metrics_history['same_crow_std'].append(val_metrics['same_crow_std'])
        metrics_history['diff_crow_mean'].append(val_metrics['diff_crow_mean'])
        metrics_history['diff_crow_std'].append(val_metrics['diff_crow_std'])
        
        # Log metrics
        logger.info(f"Epoch {epoch+1}:")
        logger.info(f"  Train Loss: {avg_train_loss:.4f}")
        logger.info(f"  Val Loss: {avg_val_loss:.4f}")
        logger.info(f"  Same Crow Similarity: {val_metrics['same_crow_mean']:.4f} ± {val_metrics['same_crow_std']:.4f}")
        logger.info(f"  Diff Crow Similarity: {val_metrics['diff_crow_mean']:.4f} ± {val_metrics['diff_crow_std']:.4f}")
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'metrics': val_metrics
        }
        torch.save(checkpoint, os.path.join(output_dir, f'checkpoint_epoch_{epoch+1}.pth'))
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), os.path.join(output_dir, 'best_model.pth'))
            logger.info("Saved new best model")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                break
    
    # Plot metrics
    plot_metrics(metrics_history, output_dir)
    
    # Save metrics history
    with open(os.path.join(output_dir, 'metrics_history.json'), 'w') as f:
        json.dump(metrics_history, f, indent=2)
    
    return model

if __name__ == "__main__":
    # Config
    CROP_DIR = 'crow_crops'  # Directory with subfolders per crow ID
    OUTPUT_DIR = 'training_output'
    BATCH_SIZE = 32
    EPOCHS = 50
    EMBED_DIM = 512
    LR = 1e-4
    VAL_SPLIT = 0.2
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Save training config
    config = {
        'crop_dir': CROP_DIR,
        'batch_size': BATCH_SIZE,
        'epochs': EPOCHS,
        'embed_dim': EMBED_DIM,
        'learning_rate': LR,
        'val_split': VAL_SPLIT,
        'device': DEVICE,
        'timestamp': datetime.now().isoformat()
    }
    with open(os.path.join(OUTPUT_DIR, 'training_config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    # Create datasets
    full_dataset = CrowTripletDataset(CROP_DIR)
    val_size = int(len(full_dataset) * VAL_SPLIT)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Model
    model = CrowResNetEmbedder(embedding_dim=EMBED_DIM).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = TripletLoss(margin=1.0, mining_type='hard')
    
    # Train
    logger.info(f"Training on {len(train_dataset)} images, validating on {len(val_dataset)} images")
    logger.info(f"Total crows: {len(full_dataset.crow_ids)}")
    model = train_model(
        model, 
        train_loader, 
        val_loader,
        criterion, 
        optimizer, 
        DEVICE, 
        EPOCHS,
        OUTPUT_DIR
    )
    
    logger.info("Training complete. Check training_output directory for results.") 