import os
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import cv2
import numpy as np
from models import CrowResNetEmbedder
from torchvision import transforms

class CrowTripletDataset(Dataset):
    def __init__(self, crop_dir, transform=None):
        """
        Dataset for training crow embeddings using triplet loss.
        Args:
            crop_dir: Directory containing subdirectories for each crow ID
            transform: Optional torchvision transforms
        """
        self.samples = []
        self.crow_to_imgs = {}
        self.transform = transform or transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Load all crow images
        for crow_id in os.listdir(crop_dir):
            crow_path = os.path.join(crop_dir, crow_id)
            if not os.path.isdir(crow_path):
                continue
                
            imgs = []
            for img_name in os.listdir(crow_path):
                if not img_name.endswith(('.jpg', '.png')):
                    continue
                    
                img_path = os.path.join(crow_path, img_name)
                try:
                    # Load and preprocess image
                    img = cv2.imread(img_path)
                    if img is None:
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
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
                    continue
            
            if len(imgs) >= 2:  # Need at least 2 images for triplet
                self.crow_to_imgs[crow_id] = imgs
                for img in imgs:
                    self.samples.append((crow_id, img))
        
        self.crow_ids = list(self.crow_to_imgs.keys())
        print(f"Loaded {len(self.samples)} images from {len(self.crow_ids)} crows")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        anchor_crow, anchor_img = self.samples[idx]
        
        # Get positive sample (different image of same crow)
        pos_img = anchor_img
        while pos_img is anchor_img:
            pos_img = np.random.choice(self.crow_to_imgs[anchor_crow])
        
        # Get negative sample (image of different crow)
        neg_crow = anchor_crow
        while neg_crow == anchor_crow:
            neg_crow = np.random.choice(self.crow_ids)
        neg_img = np.random.choice(self.crow_to_imgs[neg_crow])
        
        # Apply transforms
        if self.transform:
            anchor_img = self.transform(anchor_img)
            pos_img = self.transform(pos_img)
            neg_img = self.transform(neg_img)
        
        return anchor_img, pos_img, neg_img

def train_model(model, train_loader, criterion, optimizer, device, epochs):
    """Train the model using triplet loss."""
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for anchor, positive, negative in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
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
        
        avg_loss = total_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}")
        
        # Save checkpoint
        torch.save(model.state_dict(), f'crow_resnet_triplet_epoch_{epoch+1}.pth')

if __name__ == "__main__":
    # Config
    CROP_DIR = 'crow_crops'  # Directory with subfolders per crow ID
    BATCH_SIZE = 32
    EPOCHS = 20
    EMBED_DIM = 512  # Using 512 dimensions to match database
    LR = 1e-4
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Data augmentation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(30),  # Increased from 15° to 30°
        transforms.ColorJitter(brightness=0.3, contrast=0.3),  # Increased from 0.2 to 0.3
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Dataset and loader
    dataset = CrowTripletDataset(CROP_DIR, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    
    # Model
    model = CrowResNetEmbedder(embedding_dim=EMBED_DIM).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.TripletMarginLoss(margin=1.0, p=2)
    
    # Train
    print(f"Training on {len(dataset)} images from {len(dataset.crow_ids)} crows")
    train_model(model, dataloader, criterion, optimizer, DEVICE, EPOCHS)
    
    # Save final model
    torch.save(model.state_dict(), 'crow_resnet_triplet.pth')
    print("Training complete. Model saved to crow_resnet_triplet.pth") 