import cv2
import numpy as np
import logging
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)

class MultiViewExtractor:
    def __init__(self, 
                 rotation_angles: List[float] = [-30, 30],
                 zoom_factors: List[float] = [1.2],
                 min_size: int = 100,
                 max_size: int = 1000,
                 interpolation: int = cv2.INTER_LINEAR):
        """
        Initialize the multi-view extractor.
        
        Args:
            rotation_angles: List of angles in degrees to rotate the image
            zoom_factors: List of zoom factors (>1 for zoom in, <1 for zoom out)
            min_size: Minimum size (width/height) of the image to process
            max_size: Maximum size (width/height) of the image to process
            interpolation: OpenCV interpolation method for resizing
        """
        self.rotation_angles = rotation_angles
        self.zoom_factors = zoom_factors
        self.min_size = min_size
        self.max_size = max_size
        self.interpolation = interpolation
        logger.info(f"Initialized MultiViewExtractor with {len(rotation_angles)} rotations and {len(zoom_factors)} zoom factors")
    
    def _resize_if_needed(self, img: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Resize image if it's too large or too small.
        Returns the resized image and the scale factor used.
        """
        h, w = img.shape[:2]
        scale = 1.0
        
        # Handle too small images
        if min(h, w) < self.min_size:
            scale = self.min_size / min(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=self.interpolation)
            logger.debug(f"Upscaled image from {(h, w)} to {(new_h, new_w)}")
        
        # Handle too large images
        if max(h, w) > self.max_size:
            scale = self.max_size / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=self.interpolation)
            logger.debug(f"Downscaled image from {(h, w)} to {(new_h, new_w)}")
        
        return img, scale
    
    def _rotate_image(self, img: np.ndarray, angle: float) -> np.ndarray:
        """
        Rotate image by the specified angle.
        """
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(img, matrix, (w, h), flags=self.interpolation)
        logger.debug(f"Rotated image by {angle} degrees")
        return rotated
    
    def _zoom_image(self, img: np.ndarray, factor: float) -> np.ndarray:
        """
        Zoom image by the specified factor.
        """
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        
        # Calculate new dimensions
        new_w = int(w * factor)
        new_h = int(h * factor)
        
        # Calculate translation to keep center point
        tx = (new_w - w) // 2
        ty = (new_h - h) // 2
        
        # Create transformation matrix
        matrix = np.array([
            [factor, 0, tx],
            [0, factor, ty]
        ], dtype=np.float32)
        
        # Apply zoom
        zoomed = cv2.warpAffine(img, matrix, (new_w, new_h), flags=self.interpolation)
        logger.debug(f"Zoomed image by factor {factor}")
        return zoomed
    
    def extract(self, img: np.ndarray) -> List[np.ndarray]:
        """
        Extract multiple views from the input image.
        
        Args:
            img: Input image (BGR format)
            
        Returns:
            List of views including:
            - Original image (resized if needed)
            - Rotated versions
            - Zoomed versions
        """
        # Validate input
        if img is None:
            logger.error("Input image is None")
            return []
        if not isinstance(img, np.ndarray):
            logger.error(f"Input is not a numpy array: {type(img)}")
            return []
        if img.size == 0:
            logger.error("Input image is empty")
            return []
        if len(img.shape) != 3 or img.shape[2] != 3:
            logger.error(f"Invalid image shape: {img.shape}")
            return []
        
        try:
            # Resize if needed
            img, scale = self._resize_if_needed(img)
            views = [img]  # Start with original image
            
            # Add rotated views
            for angle in self.rotation_angles:
                rotated = self._rotate_image(img, angle)
                views.append(rotated)
            
            # Add zoomed views
            for factor in self.zoom_factors:
                zoomed = self._zoom_image(img, factor)
                views.append(zoomed)
            
            logger.info(f"Generated {len(views)} views from input image")
            return views
            
        except Exception as e:
            logger.error(f"Error in multi-view extraction: {str(e)}", exc_info=True)
            return []  # Return empty list if something goes wrong

def create_multi_view_extractor(
    rotation_angles: Optional[List[float]] = None,
    zoom_factors: Optional[List[float]] = None,
    min_size: int = 100,
    max_size: int = 1000
) -> MultiViewExtractor:
    """
    Create a multi-view extractor with the specified parameters.
    
    Args:
        rotation_angles: List of angles in degrees to rotate the image
        zoom_factors: List of zoom factors (>1 for zoom in, <1 for zoom out)
        min_size: Minimum size (width/height) of the image to process
        max_size: Maximum size (width/height) of the image to process
        
    Returns:
        MultiViewExtractor instance
    """
    if rotation_angles is None:
        rotation_angles = [-30, 30]
    if zoom_factors is None:
        zoom_factors = [1.2]
    
    return MultiViewExtractor(
        rotation_angles=rotation_angles,
        zoom_factors=zoom_factors,
        min_size=min_size,
        max_size=max_size
    ) 