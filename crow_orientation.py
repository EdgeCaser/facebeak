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
        Uses much more conservative and reliable heuristics.
        
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
                
            # 1. CONSERVATIVE: Prefer minimal rotation (original orientation is often correct)
            if rotation == 0:
                score += 1.0  # Bias toward original orientation
            elif rotation == 180:
                score += 0.5  # Second preference (just flipped)
            else:
                score -= 0.5  # Penalty for 90/270 degree rotations
                
            # 2. CENTER OF MASS ANALYSIS
            # Find the dark regions (crow body) and check their distribution
            # Threshold to find dark regions (crow body)
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Calculate center of mass of dark regions
            moments = cv2.moments(thresh)
            if moments['m00'] > 0:  # Avoid division by zero
                cx = moments['m10'] / moments['m00']  # Center X
                cy = moments['m01'] / moments['m00']  # Center Y
                
                # For upright crow, center of mass should be:
                # - Horizontally centered (cx near w/2)
                # - Vertically in lower 2/3 (cy > h/3)
                
                horizontal_centering = 1.0 - abs(cx - w/2) / (w/2)  # 1.0 = perfectly centered
                vertical_position = cy / h  # 0.0 = top, 1.0 = bottom
                
                score += horizontal_centering * 0.5  # Moderate weight
                
                # Prefer center of mass in middle-to-lower region (0.4 to 0.8)
                if 0.4 <= vertical_position <= 0.8:
                    score += 1.0
                elif 0.3 <= vertical_position <= 0.9:
                    score += 0.5
                else:
                    score -= 0.5
                    
            # 3. SHAPE COMPACTNESS
            # Find main contour and analyze its shape
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Get largest contour (should be the crow)
                main_contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(main_contour)
                
                if area > 100:  # Reasonable size
                    # Calculate bounding rectangle
                    x, y, w_rect, h_rect = cv2.boundingRect(main_contour)
                    
                    # For upright crow, height should be >= width
                    rect_aspect = h_rect / (w_rect + 1e-6)
                    
                    if rect_aspect >= 1.2:  # Clearly taller than wide
                        score += 1.5
                    elif rect_aspect >= 1.0:  # Roughly square or slightly tall
                        score += 0.5
                    elif rect_aspect >= 0.8:  # Slightly wide
                        score += 0.0
                    else:  # Very wide (probably sideways)
                        score -= 1.0
                        
                    # Calculate fill ratio (how much of bounding rect is filled)
                    fill_ratio = area / (w_rect * h_rect)
                    
                    # Crows should have reasonable fill ratio (not too sparse)
                    if 0.3 <= fill_ratio <= 0.8:
                        score += 0.5
                        
            # 4. EDGE DISTRIBUTION ANALYSIS
            # Look for strong edges that suggest correct orientation
            edges = cv2.Canny(gray, 30, 100)
            
            # Split into regions and analyze edge patterns
            top_edges = edges[:h//3, :]
            middle_edges = edges[h//3:2*h//3, :]
            bottom_edges = edges[2*h//3:, :]
            
            top_density = np.sum(top_edges) / (top_edges.size + 1e-6)
            middle_density = np.sum(middle_edges) / (middle_edges.size + 1e-6)
            bottom_density = np.sum(bottom_edges) / (bottom_edges.size + 1e-6)
            
            # For upright crow:
            # - Top should have moderate edges (head features)
            # - Middle should have strong edges (body outline)
            # - Bottom should have fewer edges (legs, ground)
            
            if middle_density > top_density and middle_density > bottom_density:
                score += 0.5  # Body region has most edges
                
            # 5. VERY CONSERVATIVE: Avoid dramatic changes unless clearly better
            # Only apply orientation correction if the alternative is CLEARLY better
            score_threshold_bonus = 0.0
            if rotation != 0:
                # Require higher confidence for non-zero rotations
                score_threshold_bonus = -0.5
                
            return score + score_threshold_bonus
            
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
                if rotation != 0 or flip != False:
                    self.logger.info(f"Applied orientation correction: rotation={rotation}°, flip={flip}")
                else:
                    self.logger.debug(f"No orientation correction needed: rotation={rotation}°, flip={flip}")
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