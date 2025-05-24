#!/usr/bin/env python3
"""
Improved Triplet Loss Implementation
Features better mining strategies and adaptive margins for crow identification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.spatial.distance import pdist, squareform


class ImprovedTripletLoss(nn.Module):
    def __init__(self, margin=1.0, mining_type='adaptive', margin_schedule=None, 
                 alpha=0.2, beta=0.02):
        """
        Improved triplet loss with multiple mining strategies.
        
        Args:
            margin: Base margin for triplet loss
            mining_type: 'hard', 'semi_hard', 'adaptive', 'curriculum'
            margin_schedule: Function to schedule margin during training
            alpha: Weight for adaptive margin calculation
            beta: Temperature parameter for soft mining
        """
        super().__init__()
        self.margin = margin
        self.mining_type = mining_type
        self.margin_schedule = margin_schedule
        self.alpha = alpha
        self.beta = beta
        self.epoch = 0
        
    def forward(self, embeddings, labels):
        """
        Compute triplet loss with improved mining.
        
        Args:
            embeddings: (N, D) tensor of embeddings
            labels: (N,) tensor of class labels
        
        Returns:
            loss: Scalar loss value
            stats: Dictionary with mining statistics
        """
        # Compute pairwise distances
        distances = self._compute_pairwise_distances(embeddings)
        
        # Get valid triplets based on labels
        anchor_positive_pairs, anchor_negative_pairs = self._get_valid_pairs(labels)
        
        # Mine triplets based on strategy
        if self.mining_type == 'hard':
            triplets, stats = self._hard_mining(distances, anchor_positive_pairs, anchor_negative_pairs)
        elif self.mining_type == 'semi_hard':
            triplets, stats = self._semi_hard_mining(distances, anchor_positive_pairs, anchor_negative_pairs)
        elif self.mining_type == 'adaptive':
            triplets, stats = self._adaptive_mining(distances, anchor_positive_pairs, anchor_negative_pairs)
        elif self.mining_type == 'curriculum':
            triplets, stats = self._curriculum_mining(distances, anchor_positive_pairs, anchor_negative_pairs)
        else:
            raise ValueError(f"Unknown mining type: {self.mining_type}")
        
        if len(triplets) == 0:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True), stats
        
        # Compute triplet loss
        loss = self._compute_triplet_loss(distances, triplets)
        
        # Update statistics
        stats['loss'] = loss.item()
        stats['n_triplets'] = len(triplets)
        stats['epoch'] = self.epoch
        
        return loss, stats
    
    def _compute_pairwise_distances(self, embeddings):
        """Compute pairwise Euclidean distances."""
        # Normalize embeddings for better numerical stability
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        # Compute squared distances
        dot_product = torch.mm(embeddings, embeddings.t())
        squared_norm = torch.diag(dot_product)
        distances = squared_norm.unsqueeze(0) - 2.0 * dot_product + squared_norm.unsqueeze(1)
        distances = torch.clamp(distances, min=0.0)  # Ensure non-negative
        
        return torch.sqrt(distances + 1e-12)  # Add small epsilon for numerical stability
    
    def _get_valid_pairs(self, labels):
        """Get valid anchor-positive and anchor-negative pairs."""
        labels = labels.cpu().numpy()
        
        anchor_positive_pairs = []
        anchor_negative_pairs = []
        
        for i in range(len(labels)):
            # Positive pairs (same class)
            positive_indices = np.where(labels == labels[i])[0]
            positive_indices = positive_indices[positive_indices != i]  # Exclude self
            for j in positive_indices:
                anchor_positive_pairs.append((i, j))
            
            # Negative pairs (different class)
            negative_indices = np.where(labels != labels[i])[0]
            for j in negative_indices:
                anchor_negative_pairs.append((i, j))
        
        return anchor_positive_pairs, anchor_negative_pairs
    
    def _hard_mining(self, distances, anchor_positive_pairs, anchor_negative_pairs):
        """Hard negative mining: select hardest positives and negatives."""
        if not anchor_positive_pairs or not anchor_negative_pairs:
            return [], {'mining_type': 'hard', 'n_hard_triplets': 0}
        
        # Group by anchor
        anchor_to_positives = {}
        anchor_to_negatives = {}
        
        for anchor, positive in anchor_positive_pairs:
            if anchor not in anchor_to_positives:
                anchor_to_positives[anchor] = []
            anchor_to_positives[anchor].append(positive)
        
        for anchor, negative in anchor_negative_pairs:
            if anchor not in anchor_to_negatives:
                anchor_to_negatives[anchor] = []
            anchor_to_negatives[anchor].append(negative)
        
        triplets = []
        for anchor in anchor_to_positives:
            if anchor not in anchor_to_negatives:
                continue
            
            positives = anchor_to_positives[anchor]
            negatives = anchor_to_negatives[anchor]
            
            # Select hardest positive (farthest)
            positive_distances = [distances[anchor, pos].item() for pos in positives]
            hardest_positive = positives[np.argmax(positive_distances)]
            
            # Select hardest negative (closest)
            negative_distances = [distances[anchor, neg].item() for neg in negatives]
            hardest_negative = negatives[np.argmin(negative_distances)]
            
            triplets.append((anchor, hardest_positive, hardest_negative))
        
        stats = {
            'mining_type': 'hard',
            'n_hard_triplets': len(triplets),
            'avg_positive_dist': np.mean([distances[a, p].item() for a, p, _ in triplets]),
            'avg_negative_dist': np.mean([distances[a, n].item() for _, _, n in triplets])
        }
        
        return triplets, stats
    
    def _semi_hard_mining(self, distances, anchor_positive_pairs, anchor_negative_pairs):
        """Semi-hard mining: select negatives closer than margin but farther than positive."""
        if not anchor_positive_pairs or not anchor_negative_pairs:
            return [], {'mining_type': 'semi_hard', 'n_semi_hard_triplets': 0}
        
        # Group by anchor
        anchor_to_positives = {}
        anchor_to_negatives = {}
        
        for anchor, positive in anchor_positive_pairs:
            if anchor not in anchor_to_positives:
                anchor_to_positives[anchor] = []
            anchor_to_positives[anchor].append(positive)
        
        for anchor, negative in anchor_negative_pairs:
            if anchor not in anchor_to_negatives:
                anchor_to_negatives[anchor] = []
            anchor_to_negatives[anchor].append(negative)
        
        triplets = []
        for anchor in anchor_to_positives:
            if anchor not in anchor_to_negatives:
                continue
            
            positives = anchor_to_positives[anchor]
            negatives = anchor_to_negatives[anchor]
            
            for positive in positives:
                pos_dist = distances[anchor, positive].item()
                
                # Find semi-hard negatives
                semi_hard_negatives = []
                for negative in negatives:
                    neg_dist = distances[anchor, negative].item()
                    # Semi-hard: pos_dist < neg_dist < pos_dist + margin
                    if pos_dist < neg_dist < pos_dist + self.margin:
                        semi_hard_negatives.append(negative)
                
                if semi_hard_negatives:
                    # Select random semi-hard negative
                    negative = np.random.choice(semi_hard_negatives)
                    triplets.append((anchor, positive, negative))
        
        stats = {
            'mining_type': 'semi_hard',
            'n_semi_hard_triplets': len(triplets)
        }
        
        return triplets, stats
    
    def _adaptive_mining(self, distances, anchor_positive_pairs, anchor_negative_pairs):
        """Adaptive mining: adjust difficulty based on embedding quality."""
        if not anchor_positive_pairs or not anchor_negative_pairs:
            return [], {'mining_type': 'adaptive', 'n_adaptive_triplets': 0}
        
        # Compute embedding quality metrics
        embedding_quality = self._compute_embedding_quality(distances, anchor_positive_pairs, anchor_negative_pairs)
        
        # Adjust margin based on quality
        adaptive_margin = self.margin * (1 + self.alpha * (1 - embedding_quality))
        
        # Group by anchor
        anchor_to_positives = {}
        anchor_to_negatives = {}
        
        for anchor, positive in anchor_positive_pairs:
            if anchor not in anchor_to_positives:
                anchor_to_positives[anchor] = []
            anchor_to_positives[anchor].append(positive)
        
        for anchor, negative in anchor_negative_pairs:
            if anchor not in anchor_to_negatives:
                anchor_to_negatives[anchor] = []
            anchor_to_negatives[anchor].append(negative)
        
        triplets = []
        for anchor in anchor_to_positives:
            if anchor not in anchor_to_negatives:
                continue
            
            positives = anchor_to_positives[anchor]
            negatives = anchor_to_negatives[anchor]
            
            for positive in positives:
                pos_dist = distances[anchor, positive].item()
                
                # Find violating negatives
                violating_negatives = []
                for negative in negatives:
                    neg_dist = distances[anchor, negative].item()
                    if neg_dist < pos_dist + adaptive_margin:
                        violating_negatives.append((negative, neg_dist))
                
                if violating_negatives:
                    # Sort by difficulty and select
                    violating_negatives.sort(key=lambda x: x[1])  # Sort by distance
                    
                    # Select based on embedding quality
                    if embedding_quality > 0.7:  # High quality: select hardest
                        negative = violating_negatives[0][0]
                    elif embedding_quality > 0.4:  # Medium quality: select random hard
                        idx = np.random.randint(0, min(3, len(violating_negatives)))
                        negative = violating_negatives[idx][0]
                    else:  # Low quality: select easier negatives
                        idx = np.random.randint(max(0, len(violating_negatives)//2), len(violating_negatives))
                        negative = violating_negatives[idx][0]
                    
                    triplets.append((anchor, positive, negative))
        
        stats = {
            'mining_type': 'adaptive',
            'n_adaptive_triplets': len(triplets),
            'embedding_quality': embedding_quality,
            'adaptive_margin': adaptive_margin
        }
        
        return triplets, stats
    
    def _curriculum_mining(self, distances, anchor_positive_pairs, anchor_negative_pairs):
        """Curriculum learning: gradually increase difficulty."""
        if not anchor_positive_pairs or not anchor_negative_pairs:
            return [], {'mining_type': 'curriculum', 'n_curriculum_triplets': 0}
        
        # Determine difficulty level based on epoch
        difficulty = min(1.0, self.epoch / 50.0)  # Gradually increase over 50 epochs
        
        # Group by anchor
        anchor_to_positives = {}
        anchor_to_negatives = {}
        
        for anchor, positive in anchor_positive_pairs:
            if anchor not in anchor_to_positives:
                anchor_to_positives[anchor] = []
            anchor_to_positives[anchor].append(positive)
        
        for anchor, negative in anchor_negative_pairs:
            if anchor not in anchor_to_negatives:
                anchor_to_negatives[anchor] = []
            anchor_to_negatives[anchor].append(negative)
        
        triplets = []
        for anchor in anchor_to_positives:
            if anchor not in anchor_to_negatives:
                continue
            
            positives = anchor_to_positives[anchor]
            negatives = anchor_to_negatives[anchor]
            
            # Select positive based on difficulty
            positive_distances = [(pos, distances[anchor, pos].item()) for pos in positives]
            positive_distances.sort(key=lambda x: x[1])  # Sort by distance
            
            # Early training: easier positives (closer), later: harder positives (farther)
            pos_idx = int((1 - difficulty) * len(positive_distances))
            pos_idx = min(pos_idx, len(positive_distances) - 1)
            positive = positive_distances[pos_idx][0]
            pos_dist = positive_distances[pos_idx][1]
            
            # Select negative based on difficulty
            negative_distances = [(neg, distances[anchor, neg].item()) for neg in negatives]
            negative_distances.sort(key=lambda x: x[1])  # Sort by distance
            
            # Filter negatives within curriculum margin
            curriculum_margin = self.margin * (0.5 + 0.5 * difficulty)
            valid_negatives = [(neg, dist) for neg, dist in negative_distances 
                             if dist < pos_dist + curriculum_margin]
            
            if valid_negatives:
                # Early training: easier negatives (farther), later: harder negatives (closer)
                neg_idx = int(difficulty * len(valid_negatives))
                neg_idx = min(neg_idx, len(valid_negatives) - 1)
                negative = valid_negatives[neg_idx][0]
                
                triplets.append((anchor, positive, negative))
        
        stats = {
            'mining_type': 'curriculum',
            'n_curriculum_triplets': len(triplets),
            'difficulty': difficulty,
            'curriculum_margin': curriculum_margin
        }
        
        return triplets, stats
    
    def _compute_embedding_quality(self, distances, anchor_positive_pairs, anchor_negative_pairs):
        """Compute overall embedding quality metric."""
        if not anchor_positive_pairs or not anchor_negative_pairs:
            return 0.0
        
        # Compute mean intra-class and inter-class distances
        intra_distances = [distances[a, p].item() for a, p in anchor_positive_pairs]
        inter_distances = [distances[a, n].item() for a, n in anchor_negative_pairs]
        
        if not intra_distances or not inter_distances:
            return 0.0
        
        mean_intra = np.mean(intra_distances)
        mean_inter = np.mean(inter_distances)
        
        # Quality metric: higher when inter > intra
        if mean_intra == 0:
            return 1.0
        
        quality = (mean_inter - mean_intra) / (mean_inter + mean_intra)
        return max(0.0, quality)
    
    def _compute_triplet_loss(self, distances, triplets):
        """Compute triplet loss for selected triplets."""
        if not triplets:
            return torch.tensor(0.0, requires_grad=True)
        
        total_loss = 0.0
        device = distances.device
        
        for anchor, positive, negative in triplets:
            ap_dist = distances[anchor, positive]
            an_dist = distances[anchor, negative]
            
            # Current margin (may be scheduled)
            current_margin = self.margin
            if self.margin_schedule:
                current_margin = self.margin_schedule(self.epoch)
            
            loss = torch.clamp(ap_dist - an_dist + current_margin, min=0.0)
            total_loss = total_loss + loss
        
        return total_loss / len(triplets)
    
    def update_epoch(self, epoch):
        """Update epoch for curriculum learning and margin scheduling."""
        self.epoch = epoch


class FocalTripletLoss(nn.Module):
    def __init__(self, margin=1.0, alpha=0.25, gamma=2.0):
        """
        Focal triplet loss: focuses on hard examples.
        
        Args:
            margin: Margin for triplet loss
            alpha: Weighting factor for hard examples
            gamma: Focusing parameter
        """
        super().__init__()
        self.margin = margin
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, anchor, positive, negative):
        """Compute focal triplet loss."""
        # Compute distances
        ap_dist = F.pairwise_distance(anchor, positive, p=2)
        an_dist = F.pairwise_distance(anchor, negative, p=2)
        
        # Standard triplet loss
        basic_loss = torch.clamp(ap_dist - an_dist + self.margin, min=0.0)
        
        # Focal weight: focus on hard examples
        focal_weight = self.alpha * (basic_loss / self.margin) ** self.gamma
        
        # Apply focal weighting
        focal_loss = focal_weight * basic_loss
        
        return focal_loss.mean()


def create_margin_schedule(schedule_type='constant', max_epochs=100, initial_margin=1.0, final_margin=0.5):
    """Create margin scheduling function."""
    if schedule_type == 'constant':
        return lambda epoch: initial_margin
    elif schedule_type == 'linear':
        return lambda epoch: initial_margin - (initial_margin - final_margin) * (epoch / max_epochs)
    elif schedule_type == 'cosine':
        return lambda epoch: final_margin + (initial_margin - final_margin) * 0.5 * (1 + np.cos(np.pi * epoch / max_epochs))
    elif schedule_type == 'exponential':
        decay_rate = np.log(final_margin / initial_margin) / max_epochs
        return lambda epoch: initial_margin * np.exp(decay_rate * epoch)
    else:
        raise ValueError(f"Unknown schedule type: {schedule_type}")


# Example usage and testing
if __name__ == '__main__':
    # Test the improved triplet loss
    torch.manual_seed(42)
    
    # Generate sample data
    batch_size = 32
    embedding_dim = 512
    n_classes = 5
    
    embeddings = torch.randn(batch_size, embedding_dim)
    labels = torch.randint(0, n_classes, (batch_size,))
    
    # Test different mining strategies
    mining_types = ['hard', 'semi_hard', 'adaptive', 'curriculum']
    
    for mining_type in mining_types:
        print(f"\nTesting {mining_type} mining:")
        
        loss_fn = ImprovedTripletLoss(margin=1.0, mining_type=mining_type)
        loss, stats = loss_fn(embeddings, labels)
        
        print(f"  Loss: {loss.item():.4f}")
        print(f"  Statistics: {stats}")
    
    # Test focal triplet loss
    print(f"\nTesting focal triplet loss:")
    anchor = torch.randn(16, embedding_dim)
    positive = torch.randn(16, embedding_dim)
    negative = torch.randn(16, embedding_dim)
    
    focal_loss_fn = FocalTripletLoss(margin=1.0, alpha=0.25, gamma=2.0)
    focal_loss = focal_loss_fn(anchor, positive, negative)
    print(f"  Focal Loss: {focal_loss.item():.4f}") 