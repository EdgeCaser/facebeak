import os
import torch
from torch.utils.data import DataLoader, Dataset, random_split
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import cv2
import numpy as np
from models import CrowMultiModalEmbedder
from torchvision import transforms
import logging
from pathlib import Path
import json
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import random
from audio import extract_audio_features
import librosa

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
    def __init__(self, crop_dir, audio_dir=None, transform=None, split='train'):
        """
        Dataset for training crow embeddings using triplet loss with both visual and audio features.
        
        Args:
            crop_dir: Directory containing crow images
            audio_dir: Directory containing corresponding audio files (optional)
            transform: Optional torchvision transforms
            split: 'train' or 'val' to determine which transforms to use
        """
        self.samples = []
        self.crow_to_imgs = {}
        self.crow_to_audio = {}
        self.split = split
        self.audio_dir = audio_dir
        logger.debug(f"Initializing dataset from {crop_dir} for {split} split")
        
        # Image transforms
        if transform is None:
            if split == 'train':
                self.transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(10),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2),
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                      std=[0.229, 0.224, 0.225])
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                      std=[0.229, 0.224, 0.225])
                ])
        else:
            self.transform = transform
            
        # Load samples
        self._load_samples(crop_dir)
        if audio_dir:
            self._load_audio_samples(audio_dir)
            
        logger.info(f"Loaded {len(self.samples)} samples from {len(self.crow_to_imgs)} crows")
        if audio_dir:
            logger.info(f"Loaded audio for {len(self.crow_to_audio)} crows")
            
    def _load_samples(self, crop_dir):
        """Load image samples from directory."""
        crop_dir = Path(crop_dir)
        
        # Handle different directory structures
        if (crop_dir / "crows").exists():
            # New structure with crows subdirectory
            crow_dirs = list((crop_dir / "crows").glob("*"))
        else:
            # Old structure or direct images
            crow_dirs = [d for d in crop_dir.glob("*") if d.is_dir()]
            
        if not crow_dirs:
            # If no subdirectories, treat all images as one crow
            crow_dirs = [crop_dir]
            
        for crow_dir in crow_dirs:
            crow_id = crow_dir.name
            img_files = list(crow_dir.glob("*.jpg")) + list(crow_dir.glob("*.png"))
            
            if not img_files:
                logger.warning(f"No images found for crow {crow_id}")
                continue
                
            self.crow_to_imgs[crow_id] = img_files
            for img_file in img_files:
                self.samples.append((img_file, crow_id))
                
    def _load_audio_samples(self, audio_dir):
        """Load audio samples from directory."""
        audio_dir = Path(audio_dir)
        
        for crow_id, img_files in self.crow_to_imgs.items():
            # Look for audio files with matching crow ID
            audio_files = list(audio_dir.glob(f"{crow_id}_*.wav")) + \
                         list(audio_dir.glob(f"{crow_id}_*.mp3"))
            
            if audio_files:
                self.crow_to_audio[crow_id] = audio_files
                logger.debug(f"Found {len(audio_files)} audio files for crow {crow_id}")
            else:
                logger.warning(f"No audio files found for crow {crow_id}")
                
    def _load_and_preprocess_audio(self, audio_path):
        """Load and preprocess audio file with data augmentation."""
        try:
            # Extract features
            mel_spec, chroma = extract_audio_features(str(audio_path))

            # Apply data augmentation if training
            if self.split == 'train':
                # Pitch shifting
                if random.random() < 0.5:
                    n_steps = random.uniform(-2, 2)
                    mel_spec = librosa.effects.pitch_shift(mel_spec, sr=22050, n_steps=n_steps)
                # Time stretching
                if random.random() < 0.5:
                    rate = random.uniform(0.8, 1.2)
                    mel_spec = librosa.effects.time_stretch(mel_spec, rate=rate)

            # Ensure normalization to [0, 1] after augmentation
            mel_min, mel_max = np.nanmin(mel_spec), np.nanmax(mel_spec)
            if mel_max - mel_min > 1e-8:
                mel_spec = (mel_spec - mel_min) / (mel_max - mel_min)
            else:
                mel_spec = np.zeros_like(mel_spec)
            chroma_min, chroma_max = np.nanmin(chroma), np.nanmax(chroma)
            if chroma_max - chroma_min > 1e-8:
                chroma = (chroma - chroma_min) / (chroma_max - chroma_min)
            else:
                chroma = np.zeros_like(chroma)

            # Convert to tensor
            mel_spec = torch.from_numpy(mel_spec).float()
            chroma = torch.from_numpy(chroma).float()
        
            # Dynamic padding or truncation based on actual length
            target_time = min(max(64, mel_spec.size(1)), 256)  # Adaptive target between 64 and 256
            
            # Handle mel spectrogram
            if mel_spec.size(1) > target_time:
                mel_spec = mel_spec[:, :target_time]
            elif mel_spec.size(1) < target_time:
                padding = torch.zeros(mel_spec.size(0), target_time - mel_spec.size(1))
                mel_spec = torch.cat([mel_spec, padding], dim=1)
            
            # Handle chroma
            if chroma.size(1) > target_time:
                chroma = chroma[:, :target_time]
            elif chroma.size(1) < target_time:
                padding = torch.zeros(chroma.size(0), target_time - chroma.size(1))
                chroma = torch.cat([chroma, padding], dim=1)
            
            return {
                'mel_spec': mel_spec,
                'chroma': chroma
            }
        
        except Exception as e:
            logger.error(f"Error processing audio file {audio_path}: {e}")
            return None
            
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        img_path, crow_id = self.samples[idx]
        
        # Load and transform image
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transform(img)
        
        # Get audio features if available
        audio_features = None
        if self.audio_dir and crow_id in self.crow_to_audio:
            # Randomly select an audio file for this crow
            audio_path = random.choice(self.crow_to_audio[crow_id])
            audio_features = self._load_and_preprocess_audio(audio_path)
            
        return img, audio_features, crow_id
        
    def get_triplet(self, idx):
        """Get a triplet of samples (anchor, positive, negative)."""
        anchor_img, anchor_audio, anchor_id = self[idx]
        
        # Get positive sample (same crow)
        positive_idx = random.choice([i for i, (_, _, cid) in enumerate(self.samples) 
                                    if cid == anchor_id and i != idx])
        positive_img, positive_audio, _ = self[positive_idx]
        
        # Get negative sample (different crow)
        negative_idx = random.choice([i for i, (_, _, cid) in enumerate(self.samples) 
                                    if cid != anchor_id])
        negative_img, negative_audio, _ = self[negative_idx]
        
        return (anchor_img, positive_img, negative_img), \
               (anchor_audio, positive_audio, negative_audio), \
               anchor_id

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
    """Compute validation metrics for multi-modal model."""
    model.eval()
    all_embeddings = []
    all_labels = []
    
    with torch.no_grad():
        for (anchor_imgs, pos_imgs, neg_imgs), (anchor_audio, pos_audio, neg_audio), label in val_loader:
            # Move to device
            anchor_imgs = anchor_imgs.to(device)
            pos_imgs = pos_imgs.to(device)
            neg_imgs = neg_imgs.to(device)
            
            if anchor_audio is not None:
                anchor_audio = anchor_audio.to(device)
                pos_audio = pos_audio.to(device)
                neg_audio = neg_audio.to(device)
            
            # Get embeddings
            anchor_emb = model(anchor_imgs, anchor_audio).cpu().numpy()
            pos_emb = model(pos_imgs, pos_audio).cpu().numpy()
            neg_emb = model(neg_imgs, neg_audio).cpu().numpy()
            
            all_embeddings.extend([anchor_emb, pos_emb, neg_emb])
            if isinstance(label, (list, tuple)):
                all_labels.extend(label)
                all_labels.extend(label)
                all_labels.extend(label)
            else:
                all_labels.extend([label, label, label])
                
    # Rest of the function remains the same
    all_embeddings = np.vstack(all_embeddings)
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
    """Train the multi-modal model using triplet loss."""
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
        for (anchor_imgs, pos_imgs, neg_imgs), (anchor_audio, pos_audio, neg_audio), _ in tqdm(train_loader, 
                                                                                              desc=f"Epoch {epoch+1}/{epochs}"):
            # Move to device
            anchor_imgs = anchor_imgs.to(device)
            pos_imgs = pos_imgs.to(device)
            neg_imgs = neg_imgs.to(device)
            
            if anchor_audio is not None:
                anchor_audio = anchor_audio.to(device)
                pos_audio = pos_audio.to(device)
                neg_audio = neg_audio.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            anchor_emb = model(anchor_imgs, anchor_audio)
            pos_emb = model(pos_imgs, pos_audio)
            neg_emb = model(neg_imgs, neg_audio)
            
            # Compute triplet loss
            loss = criterion(anchor_emb, pos_emb, neg_emb)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * anchor_imgs.size(0)
            
        avg_train_loss = total_loss / len(train_loader.dataset)
        metrics_history['train_loss'].append(avg_train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for (anchor_imgs, pos_imgs, neg_imgs), (anchor_audio, pos_audio, neg_audio), _ in val_loader:
                anchor_imgs = anchor_imgs.to(device)
                pos_imgs = pos_imgs.to(device)
                neg_imgs = neg_imgs.to(device)
                
                if anchor_audio is not None:
                    anchor_audio = anchor_audio.to(device)
                    pos_audio = pos_audio.to(device)
                    neg_audio = neg_audio.to(device)
                
                anchor_emb = model(anchor_imgs, anchor_audio)
                pos_emb = model(pos_imgs, pos_audio)
                neg_emb = model(neg_imgs, neg_audio)
                
                loss = criterion(anchor_emb, pos_emb, neg_emb)
                val_loss += loss.item() * anchor_imgs.size(0)
        
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
        
        # Clear GPU memory after each epoch
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    
    # Plot metrics
    plot_metrics(metrics_history, output_dir)
    
    # Save metrics history
    with open(os.path.join(output_dir, 'metrics_history.json'), 'w') as f:
        json.dump(metrics_history, f, indent=2)
    
    return model

def main():
    # Config
    CROP_DIR = 'crow_crops'  # Directory with subfolders per crow ID
    AUDIO_DIR = 'crow_audio'  # Directory with audio files
    OUTPUT_DIR = 'training_output'
    BATCH_SIZE = 32
    EPOCHS = 100
    VAL_SPLIT = 0.2
    EMBED_DIM = 512
    LR = 1e-4
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Save config
    config = {
        'crop_dir': CROP_DIR,
        'audio_dir': AUDIO_DIR,
        'batch_size': BATCH_SIZE,
        'epochs': EPOCHS,
        'val_split': VAL_SPLIT,
        'embed_dim': EMBED_DIM,
        'learning_rate': LR,
        'device': str(DEVICE),
        'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S')
    }
    with open(os.path.join(OUTPUT_DIR, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    # Create datasets
    logger.info("Initializing datasets...")
    full_dataset = CrowTripletDataset(CROP_DIR, AUDIO_DIR)
    val_size = int(len(full_dataset) * VAL_SPLIT)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
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
    logger.info("Initializing model...")
    model = CrowMultiModalEmbedder(
        visual_embed_dim=EMBED_DIM // 2,  # Split embedding dimension between modalities
        audio_embed_dim=EMBED_DIM // 2,
        final_embed_dim=EMBED_DIM
    ).to(DEVICE)
    
    # Loss and optimizer
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = TripletLoss(margin=1.0, mining_type='hard')
    
    # Train
    logger.info(f"Training on {len(train_dataset)} samples, validating on {len(val_dataset)} samples")
    logger.info(f"Total crows: {len(full_dataset.crow_to_imgs)}")
    if AUDIO_DIR:
        logger.info(f"Crows with audio: {len(full_dataset.crow_to_audio)}")
    
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
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'final_model.pth'))
    logger.info("Training complete!")
    
    # Clear GPU memory
    if DEVICE.type == 'cuda':
        torch.cuda.empty_cache()

if __name__ == '__main__':
    main() 