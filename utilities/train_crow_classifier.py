#!/usr/bin/env python3
"""
Train a crow classification model using labeled data from the database.
This creates a quality filter to clean the dataset before triplet training.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import os
import logging
from PIL import Image
import numpy as np
from collections import Counter
import json

# Import your database functions
from db import get_all_labeled_images

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CrowDataset(Dataset):
    """Dataset for labeled crow images"""
    
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
        # Label mapping
        self.label_to_idx = {
            'crow': 0,
            'not_a_crow': 1, 
            'multi_crow': 2,
            'bad_crow': 1  # Treat bad_crow same as not_a_crow
        }
        self.idx_to_label = {v: k for k, v in self.label_to_idx.items()}
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label_str = self.labels[idx]
        
        # Load and process image
        try:
            image = Image.open(image_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
                
            # Convert label to index
            label_idx = self.label_to_idx[label_str]
            
            return image, label_idx
            
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            # Return a black image and unknown label as fallback
            black_image = torch.zeros(3, 512, 512)
            return black_image, 1  # Default to not_a_crow

def create_model(num_classes=3):
    """Create ResNet18 model for classification"""
    model = models.resnet18(pretrained=True)
    
    # Replace final layer for our number of classes
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    return model

def get_transforms():
    """Get data augmentation transforms"""
    train_transform = transforms.Compose([
        transforms.Resize((580, 580)),  # Resize to larger square to preserve aspect ratio
        transforms.CenterCrop((512, 512)),  # Crop to target size
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((580, 580)),  # Resize to larger square to preserve aspect ratio
        transforms.CenterCrop((512, 512)),  # Crop to target size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def load_labeled_data():
    """Load all labeled images from unified dataset structure"""
    logger.info("Loading labeled images from unified dataset...")

    image_paths = []
    labels = []

    # Load crow images
    crow_dir = os.path.join("dataset", "crows", "generic")
    if os.path.exists(crow_dir):
        for fname in os.listdir(crow_dir):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                image_paths.append(os.path.join(crow_dir, fname))
                labels.append("crow")
        logger.info(f"Loaded {len([l for l in labels if l == 'crow'])} crow images from {crow_dir}")
    else:
        logger.warning(f"Crow directory not found: {crow_dir}")

    # Load non-crow images
    non_crow_dir = os.path.join("dataset", "not_crow")
    if os.path.exists(non_crow_dir):
        for fname in os.listdir(non_crow_dir):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                image_paths.append(os.path.join(non_crow_dir, fname))
                labels.append("not_a_crow")
        logger.info(f"Loaded {len([l for l in labels if l == 'not_a_crow'])} non-crow images from {non_crow_dir}")
    else:
        logger.warning(f"Non-crow directory not found: {non_crow_dir}")

    logger.info(f"Total dataset: {len(image_paths)} images")
    label_counts = Counter(labels)
    logger.info(f"Label distribution: {dict(label_counts)}")
    return image_paths, labels

def train_model(model, train_loader, val_loader, num_epochs=50, device='cuda'):
    """Train the classification model"""
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
    
    model = model.to(device)
    
    best_val_acc = 0.0
    best_model_state = None
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            if batch_idx % 10 == 0:
                logger.info(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        train_acc = 100. * train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = 100. * val_correct / val_total
        
        logger.info(f'Epoch {epoch+1}/{num_epochs}:')
        logger.info(f'  Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%')
        logger.info(f'  Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.2f}%')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            logger.info(f'  New best validation accuracy: {best_val_acc:.2f}%')
        
        scheduler.step()
    
    # Load best model
    model.load_state_dict(best_model_state)
    return model, best_val_acc

def evaluate_model(model, test_loader, device='cuda'):
    """Evaluate model and print detailed metrics"""
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Convert indices back to label names
    label_names = ['crow', 'not_a_crow', 'multi_crow']
    
    # Print classification report
    logger.info("\nClassification Report:")
    print(classification_report(all_labels, all_predictions, target_names=label_names))
    
    # Print confusion matrix
    logger.info("\nConfusion Matrix:")
    cm = confusion_matrix(all_labels, all_predictions)
    print(cm)
    
    return all_predictions, all_labels

def save_model(model, filepath='crow_classifier.pth'):
    """Save the trained model"""
    model_info = {
        'model_state_dict': model.state_dict(),
        'label_to_idx': {'crow': 0, 'not_a_crow': 1, 'multi_crow': 2},
        'model_architecture': 'resnet18'
    }
    
    torch.save(model_info, filepath)
    logger.info(f"Model saved to {filepath}")

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load data
    image_paths, labels = load_labeled_data()
    
    if len(image_paths) < 10:
        logger.error("Not enough labeled data to train. Need at least 10 images.")
        return
    
    # Split data
    train_paths, test_paths, train_labels, test_labels = train_test_split(
        image_paths, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Further split training into train/val
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_paths, train_labels, test_size=0.2, random_state=42, stratify=train_labels
    )
    
    logger.info(f"Training set: {len(train_paths)} images")
    logger.info(f"Validation set: {len(val_paths)} images") 
    logger.info(f"Test set: {len(test_paths)} images")
    
    # Get transforms
    train_transform, val_transform = get_transforms()
    
    # Create datasets
    train_dataset = CrowDataset(train_paths, train_labels, train_transform)
    val_dataset = CrowDataset(val_paths, val_labels, val_transform)
    test_dataset = CrowDataset(test_paths, test_labels, val_transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # Create model
    model = create_model(num_classes=3)
    
    # Train model
    logger.info("Starting training...")
    trained_model, best_val_acc = train_model(model, train_loader, val_loader, device=device)
    
    # Evaluate on test set
    logger.info("Evaluating on test set...")
    evaluate_model(trained_model, test_loader, device=device)
    
    # Save model
    save_model(trained_model)
    
    logger.info(f"Training completed! Best validation accuracy: {best_val_acc:.2f}%")
    logger.info("Model saved as 'crow_classifier.pth'")
    logger.info("\nNext steps:")
    logger.info("1. Use this model to filter your full dataset")
    logger.info("2. Run triplet training on the cleaned dataset")

if __name__ == "__main__":
    main() 