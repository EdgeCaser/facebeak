#!/usr/bin/env python3
"""
Crow Orientation Detection and Correction

This module provides functions to detect and correct the orientation of crow crops
to ensure consistent upright orientation for better user experience during manual review.
"""

import cv2
import numpy as np
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

class CrowOrientationDetector:
    """Detects and corrects orientation of crow crops for consistent presentation."""
    
    def __init__(self):
        """Initialize the crow orientation detector."""
        self.logger = logging.getLogger(__name__)
        
    def detect_crow_orientation(self, crop: np.ndarray) -> Tuple[int, bool]:
        """
        Detect the best orientation for a crow crop.
        
        Args:
            crop: Input crop as numpy array (H, W, C)
            
        Returns:
            Tuple of (rotation_angle, should_flip_horizontal)
            rotation_angle: 0, 90, 180, or 270 degrees
            should_flip_horizontal: whether to flip horizontally after rotation
        """
        try:
            if crop is None or crop.size == 0:
                return 0, False
                
            best_rotation = 0
            best_flip = False
            best_score = -1
            
            # Try each rotation
            for rotation in [0, 90, 180, 270]:
                for flip in [False, True]:
                    score = self._score_orientation(crop, rotation, flip)
                    if score > best_score:
                        best_score = score
                        best_rotation = rotation
                        best_flip = flip
            
            return best_rotation, best_flip
            
        except Exception as e:
            self.logger.warning(f"Error detecting crow orientation: {e}")
            return 0, False
    
    def _score_orientation(self, crop: np.ndarray, rotation: int, flip: bool) -> float:
        """
        Score how good a particular orientation is for a crow crop.
        
        Args:
            crop: Input crop
            rotation: Rotation angle (0, 90, 180, 270)
            flip: Whether to flip horizontally
            
        Returns:
            Score (higher is better)
        """
        try:
            # Apply transformation
            oriented_crop = self._apply_orientation(crop, rotation, flip)
            
            if oriented_crop is None:
                return -1
                
            h, w = oriented_crop.shape[:2]
            score = 0.0
            
            # Convert to grayscale for analysis
            if len(oriented_crop.shape) == 3:
                gray = cv2.cvtColor(oriented_crop, cv2.COLOR_BGR2GRAY)
            else:
                gray = oriented_crop
                
            # 1. Body axis analysis - crows should be taller than wide when upright
            aspect_ratio = h / w
            if aspect_ratio > 1.0:  # Taller than wide
                score += 2.0 * min(aspect_ratio / 1.5, 2.0)  # Cap the bonus
            else:
                score -= 1.0  # Penalty for being wider than tall
                
            # 2. Mass distribution - crow's body mass should be in lower 2/3
            upper_third = gray[:h//3, :]
            middle_third = gray[h//3:2*h//3, :]
            lower_third = gray[2*h//3:, :]
            
            upper_mass = np.mean(upper_third)
            middle_mass = np.mean(middle_third)
            lower_mass = np.mean(lower_third)
            
            # Body should have more mass in middle and lower thirds
            if lower_mass > upper_mass:
                score += 1.0
            if middle_mass > upper_mass:
                score += 0.5
                
            # 3. Edge orientation analysis
            edges = cv2.Canny(gray, 50, 150)
            
            # Calculate vertical vs horizontal edge strength
            sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            
            vertical_strength = np.mean(np.abs(sobel_y))
            horizontal_strength = np.mean(np.abs(sobel_x))
            
            # Crows should have strong vertical edges (body outline, legs)
            if vertical_strength > horizontal_strength:
                score += 1.0
                
            # 4. Symmetry analysis - check for vertical symmetry
            left_half = gray[:, :w//2]
            right_half = cv2.flip(gray[:, w//2:], 1)  # Flip right half
            
            # Resize to match in case of odd width
            min_width = min(left_half.shape[1], right_half.shape[1])
            left_half = left_half[:, :min_width]
            right_half = right_half[:, :min_width]
            
            # Calculate symmetry score
            if left_half.shape == right_half.shape:
                symmetry = cv2.matchTemplate(left_half, right_half, cv2.TM_CCOEFF_NORMED)[0, 0]
                score += symmetry * 0.5  # Moderate weight for symmetry
                
            # 5. Head region analysis - head should be in upper portion
            head_region = gray[:h//3, :]  # Top third
            head_variance = np.var(head_region)
            
            # Head region should have features (higher variance)
            if head_variance > np.var(gray) * 0.5:
                score += 0.5
                
            # 6. Leg/bottom analysis - bottom should have thin features (legs)
            bottom_strip = gray[int(0.8*h):, :]  # Bottom 20%
            if bottom_strip.size > 0:
                # Look for thin vertical features that could be legs
                bottom_edges = cv2.Canny(bottom_strip, 50, 150)
                leg_features = np.sum(bottom_edges) / bottom_strip.size
                score += min(leg_features * 10, 1.0)  # Normalize and cap
                
            return score
            
        except Exception as e:
            self.logger.warning(f"Error scoring orientation: {e}")
            return -1
    
    def _apply_orientation(self, crop: np.ndarray, rotation: int, flip: bool) -> Optional[np.ndarray]:
        """
        Apply rotation and flip to a crop.
        
        Args:
            crop: Input crop
            rotation: Rotation angle (0, 90, 180, 270)
            flip: Whether to flip horizontally
            
        Returns:
            Transformed crop or None if error
        """
        try:
            result = crop.copy()
            
            # Apply rotation
            if rotation == 90:
                result = cv2.rotate(result, cv2.ROTATE_90_CLOCKWISE)
            elif rotation == 180:
                result = cv2.rotate(result, cv2.ROTATE_180)
            elif rotation == 270:
                result = cv2.rotate(result, cv2.ROTATE_90_COUNTERCLOCKWISE)
            
            # Apply horizontal flip
            if flip:
                result = cv2.flip(result, 1)
                
            return result
            
        except Exception as e:
            self.logger.warning(f"Error applying orientation: {e}")
            return None
    
    def correct_crop_orientation(self, crop: np.ndarray) -> np.ndarray:
        """
        Detect and correct the orientation of a crow crop.
        
        Args:
            crop: Input crop as numpy array
            
        Returns:
            Corrected crop with proper orientation
        """
        try:
            if crop is None or crop.size == 0:
                return crop
                
            # Detect best orientation
            rotation, flip = self.detect_crow_orientation(crop)
            
            # Apply correction
            corrected = self._apply_orientation(crop, rotation, flip)
            
            if corrected is not None:
                self.logger.debug(f"Applied orientation correction: rotation={rotation}°, flip={flip}")
                return corrected
            else:
                self.logger.warning("Failed to apply orientation correction, returning original")
                return crop
                
        except Exception as e:
            self.logger.error(f"Error correcting crop orientation: {e}")
            return crop

# Global instance for easy use
crow_orientation_detector = CrowOrientationDetector()

def correct_crow_crop_orientation(crop: np.ndarray) -> np.ndarray:
    """
    Convenience function to correct crow crop orientation.
    
    Args:
        crop: Input crop as numpy array
        
    Returns:
        Corrected crop with proper orientation
    """
    return crow_orientation_detector.correct_crop_orientation(crop)

def detect_crow_orientation(crop: np.ndarray) -> Tuple[int, bool]:
    """
    Convenience function to detect crow orientation.
    
    Args:
        crop: Input crop as numpy array
        
    Returns:
        Tuple of (rotation_angle, should_flip_horizontal)
    """
    return crow_orientation_detector.detect_crow_orientation(crop) 