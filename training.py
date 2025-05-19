import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, accuracy_score, precision_score, recall_score, f1_score
import logging
from tqdm import tqdm
from torch.utils.data import DataLoader
from typing import Dict, Tuple, Optional, Union

logger = logging.getLogger(__name__)

def triplet_loss(anchor, positive, negative, margin=1.0):
    """
    Compute triplet loss.
    
    Args:
        anchor: Anchor embeddings
        positive: Positive sample embeddings
        negative: Negative sample embeddings
        margin: Margin for triplet loss
    
    Returns:
        Loss value
    """
    # Compute distances
    pos_dist = F.pairwise_distance(anchor, positive)
    neg_dist = F.pairwise_distance(anchor, negative)
    
    # Compute loss
    loss = torch.clamp(pos_dist - neg_dist + margin, min=0.0)
    
    # Add small epsilon to avoid zero loss when distances are equal
    loss = loss + 1e-8
    
    return loss.mean()

def compute_metrics(
    model: torch.nn.Module,
    data: Union[DataLoader, Dict[str, torch.Tensor]],
    device: torch.device
) -> Tuple[Dict[str, float], np.ndarray]:
    """Compute metrics for model evaluation.
    
    Args:
        model: Model to evaluate
        data: Either a DataLoader or a dictionary containing 'image', 'audio', and 'crow_id'
        device: Device to run evaluation on
        
    Returns:
        Tuple of (metrics dict, similarities array)
    """
    model.eval()
    embeddings = []
    labels = []
    
    with torch.no_grad():
        if isinstance(data, dict):
            images = data['image'].to(device)
            # Early return if batch is empty
            if images.shape[0] == 0:
                return {
                    'accuracy': 0.0,
                    'precision': 0.0,
                    'recall': 0.0,
                    'f1': 0.0
                }, np.array([])
            audio = None
            if data['audio'] is not None:
                audio = {
                    'mel_spec': data['audio']['mel_spec'].to(device),
                    'chroma': data['audio']['chroma'].to(device)
                }
            labels = data['crow_id']
            embeddings = model(images, audio).cpu().numpy()
        else:
            # Handle DataLoader format
            for batch in data:
                if isinstance(batch, dict):
                    images = batch['image'].to(device)
                    audio = None
                    if batch['audio'] is not None:
                        audio = {
                            'mel_spec': batch['audio']['mel_spec'].to(device),
                            'chroma': batch['audio']['chroma'].to(device)
                        }
                    batch_labels = batch['crow_id']
                else:
                    images, audio, batch_labels = batch
                    images = images.to(device)
                    if audio is not None:
                        audio = {
                            'mel_spec': audio[0].to(device),
                            'chroma': audio[1].to(device)
                        }
                
                batch_embeddings = model(images, audio).cpu().numpy()
                embeddings.append(batch_embeddings)
                labels.extend(batch_labels)
            
            if embeddings:
                embeddings = np.vstack(embeddings)
    
    if len(embeddings) == 0:
        return {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0
        }, np.array([])
    
    # Compute cosine similarities
    # Normalize embeddings to unit length to ensure proper cosine similarity
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    similarities = np.dot(embeddings, embeddings.T)
    
    # Ensure similarities are in [-1, 1] range
    similarities = np.clip(similarities, -1.0, 1.0)
    
    # Convert labels to integers for metric computation
    unique_labels = sorted(set(labels))
    label_to_int = {label: i for i, label in enumerate(unique_labels)}
    labels_int = np.array([label_to_int[label] for label in labels])
    
    # Compute metrics using similarity threshold of 0.5
    threshold = 0.5
    predictions = (similarities > threshold).astype(int)
    
    # Create binary labels for each pair
    true_labels = (labels_int[:, None] == labels_int[None, :]).astype(int)
    
    # Compute metrics
    accuracy = np.mean(predictions == true_labels)
    precision = np.sum((predictions == 1) & (true_labels == 1)) / (np.sum(predictions == 1) + 1e-8)
    recall = np.sum((predictions == 1) & (true_labels == 1)) / (np.sum(true_labels == 1) + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    
    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1)
    }
    
    return metrics, similarities

def training_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: Dict[str, torch.Tensor],
    device: torch.device
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Perform a single training step.
    
    Args:
        model: Model to train
        optimizer: Optimizer to use
        batch: Either a dictionary containing 'image', 'audio', and 'crow_id',
              or a tuple of (images, audio, labels)
        device: Device to run training on
        
    Returns:
        Tuple of (loss tensor, metrics dict)
    """
    model.train()
    optimizer.zero_grad()
    
    # Get data
    if isinstance(batch, dict):
        images = batch['image'].to(device)
        audio = None
        if batch['audio'] is not None:
            audio = {
                'mel_spec': batch['audio']['mel_spec'].to(device),
                'chroma': batch['audio']['chroma'].to(device)
            }
        labels = batch['crow_id']
    else:
        # Handle tuple format (imgs, audio, labels)
        images, audio, labels = batch
        images = images.to(device)
        if audio is not None:
            audio = {
                'mel_spec': audio[0].to(device),
                'chroma': audio[1].to(device)
            }
    
    # Forward pass
    embeddings = model(images, audio)
    
    # Create triplets
    unique_labels = sorted(set(labels))
    if len(unique_labels) < 2:
        # If we don't have enough unique classes, create synthetic triplets
        # by using the same sample as both anchor and positive
        triplets = []
        for i, (emb, label) in enumerate(zip(embeddings, labels)):
            # Find a negative sample (different class)
            neg_indices = [j for j, l in enumerate(labels) if l != label]
            if not neg_indices:
                # If no negative samples, use the same sample as negative
                # but with a small perturbation
                neg_emb = emb + torch.randn_like(emb) * 0.1
                triplets.append((emb, emb, neg_emb))
            else:
                # Randomly select a negative sample
                neg_idx = np.random.choice(neg_indices)
                triplets.append((emb, emb, embeddings[neg_idx]))
    else:
        # Normal triplet creation with at least two unique classes
        triplets = []
        for i, (emb, label) in enumerate(zip(embeddings, labels)):
            # Find positive sample (same class)
            pos_indices = [j for j, l in enumerate(labels) if l == label and j != i]
            if not pos_indices:
                # If no positive samples, use the same sample
                pos_emb = emb
            else:
                # Randomly select a positive sample
                pos_idx = np.random.choice(pos_indices)
                pos_emb = embeddings[pos_idx]
            
            # Find negative sample (different class)
            neg_indices = [j for j, l in enumerate(labels) if l != label]
            if not neg_indices:
                # If no negative samples, use a perturbed version of the same sample
                neg_emb = emb + torch.randn_like(emb) * 0.1
            else:
                # Randomly select a negative sample
                neg_idx = np.random.choice(neg_indices)
                neg_emb = embeddings[neg_idx]
            
            triplets.append((emb, pos_emb, neg_emb))
    
    # Compute loss
    anchors, positives, negatives = zip(*triplets)
    loss = compute_triplet_loss(
        torch.stack(anchors),
        torch.stack(positives),
        torch.stack(negatives),
        margin=1.0
    )
    
    # Backward pass
    loss.backward()
    optimizer.step()
    
    # Compute metrics
    with torch.no_grad():
        metrics, _ = compute_metrics(model, batch, device)
    
    return loss, metrics

def validation_step(
    model: torch.nn.Module,
    batch: Dict[str, torch.Tensor],
    device: torch.device
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Perform a single validation step.
    
    Args:
        model: Model to evaluate
        batch: Dictionary containing 'image', 'audio', and 'crow_id'
        device: Device to run validation on
        
    Returns:
        Tuple of (loss tensor, metrics dict)
    """
    model.eval()
    
    with torch.no_grad():
        # Get data
        images = batch['image'].to(device)
        audio = None
        if batch['audio'] is not None:
            audio = {
                'mel_spec': batch['audio']['mel_spec'].to(device),
                'chroma': batch['audio']['chroma'].to(device)
            }
        
        # Convert labels to strings
        if isinstance(batch['crow_id'], (list, tuple)):
            labels = [str(l) for l in batch['crow_id']]
        else:
            labels = [str(batch['crow_id'])]
        
        # Forward pass
        embeddings = model(images, audio)
        
        # Create triplets
        unique_labels = sorted(set(labels))
        if len(unique_labels) < 2:
            return torch.tensor(0.0, device=device), {
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'same_crow_mean': 0.0,
                'same_crow_std': 0.0,
                'diff_crow_mean': 0.0,
                'diff_crow_std': 0.0
            }
        
        triplets = []
        for i, (emb, label) in enumerate(zip(embeddings, labels)):
            pos_indices = [j for j, l in enumerate(labels) if l == label and j != i]
            neg_indices = [j for j, l in enumerate(labels) if l != label]
            
            if not pos_indices or not neg_indices:
                continue
            
            pos_idx = np.random.choice(pos_indices)
            neg_idx = np.random.choice(neg_indices)
            
            triplets.append((emb, embeddings[pos_idx], embeddings[neg_idx]))
        
        if not triplets:
            return torch.tensor(0.0, device=device), {
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'same_crow_mean': 0.0,
                'same_crow_std': 0.0,
                'diff_crow_mean': 0.0,
                'diff_crow_std': 0.0
            }
        
        # Compute loss
        anchors, positives, negatives = zip(*triplets)
        loss = compute_triplet_loss(
            torch.stack(anchors),
            torch.stack(positives),
            torch.stack(negatives),
            margin=1.0
        )
        
        # Compute metrics
        metrics, _ = compute_metrics(model, batch, device)
        
        return loss, metrics

def compute_triplet_loss(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor,
    margin: float = 1.0
) -> torch.Tensor:
    """Compute triplet loss for a batch of embeddings.
    
    Args:
        anchor: Anchor embeddings of shape (batch_size, embed_dim)
        positive: Positive embeddings of shape (batch_size, embed_dim)
        negative: Negative embeddings of shape (batch_size, embed_dim)
        margin: Margin for triplet loss
        
    Returns:
        Scalar tensor containing the triplet loss
    """
    # Compute distances using pairwise_distance
    pos_dist = F.pairwise_distance(anchor, positive)
    neg_dist = F.pairwise_distance(anchor, negative)
    
    # Compute triplet loss with margin
    # Use a small epsilon only for numerical stability in the clamp
    # This ensures we don't get exactly zero loss when distances are equal
    loss = torch.clamp(pos_dist - neg_dist + margin, min=0.0)
    
    # Add a small epsilon only to non-zero losses to avoid numerical issues
    # This preserves the zero loss case while preventing numerical instability
    mask = loss > 0
    loss = torch.where(mask, loss + 1e-8, loss)
    
    return loss.mean() 