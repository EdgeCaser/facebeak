import numpy as np
import cv2
import torch
import logging
from typing import Optional, Union, Tuple

logger = logging.getLogger(__name__)

class ColorNormalizer:
    """Base class for color normalization."""
    def normalize(self, img: np.ndarray) -> np.ndarray:
        """Normalize an image.
        
        Args:
            img: Input image as numpy array of shape (H, W, 3) with dtype uint8
            
        Returns:
            Normalized image as numpy array of same shape and dtype
        """
        raise NotImplementedError

class DummyNormalizer(ColorNormalizer):
    """Dummy normalizer that returns the input image unchanged."""
    def normalize(self, img: np.ndarray) -> np.ndarray:
        if not isinstance(img, np.ndarray):
            raise AttributeError("Input must be a numpy array")
        if img.size == 0:
            raise IndexError("Input array is empty")
        if len(img.shape) != 3 or img.shape[2] != 3:
            raise IndexError("Input must be a 3-channel image")
        if img.dtype != np.uint8:
            raise TypeError("Input must be uint8")
        return img

class AdaptiveNormalizer(ColorNormalizer):
    """Adaptive color normalizer that performs histogram equalization and contrast enhancement."""
    def __init__(self, use_gpu: bool = True):
        """Initialize the normalizer.
        
        Args:
            use_gpu: Whether to use GPU acceleration. Defaults to True.
                    Will fall back to CPU if GPU is unavailable.
        """
        self.use_gpu = use_gpu and torch.cuda.is_available()
        if self.use_gpu:
            logger.info("Using GPU acceleration for color normalization")
            # Initialize CUDA context and warm up
            try:
                dummy_tensor = torch.zeros(1, device='cuda')
                del dummy_tensor
                torch.cuda.empty_cache()
            except Exception as e:
                logger.warning(f"GPU initialization failed, falling back to CPU: {str(e)}")
                self.use_gpu = False
        else:
            logger.info("Using CPU for color normalization")
    
    def _normalize_cpu(self, img: np.ndarray) -> np.ndarray:
        """Normalize image on CPU."""
        # Convert to LAB color space
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        
        # Split channels
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        # Merge channels
        lab = cv2.merge([l, a, b])
        
        # Convert back to BGR
        normalized = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # Apply gamma correction
        gamma = 1.2
        lookup_table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255
                                for i in np.arange(0, 256)]).astype("uint8")
        normalized = cv2.LUT(normalized, lookup_table)
        
        return normalized
    
    def _normalize_gpu(self, img: np.ndarray) -> np.ndarray:
        """Normalize image on GPU."""
        # Convert to tensor
        img_tensor = torch.from_numpy(img).float().cuda()
        
        # Convert to LAB color space
        # Note: OpenCV's color conversion is more accurate, so we do this on CPU
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        lab_tensor = torch.from_numpy(lab).float().cuda()
        
        # Split channels
        l, a, b = torch.split(lab_tensor, 1, dim=2)
        
        # Apply adaptive histogram equalization to L channel
        # We use a simple approximation of CLAHE on GPU
        l = l / 255.0  # Normalize to [0, 1]
        l_mean = torch.mean(l)
        l_std = torch.std(l)
        l = (l - l_mean) / (l_std + 1e-6)
        l = torch.clamp(l * 0.5 + 0.5, 0, 1)  # Scale to [0, 1]
        l = (l * 255).to(torch.uint8)
        
        # Merge channels
        lab_tensor = torch.cat([l, a, b], dim=2)
        
        # Convert back to BGR (on CPU)
        lab = lab_tensor.cpu().numpy().astype(np.uint8)
        normalized = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # Apply gamma correction
        gamma = 1.2
        lookup_table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255
                                for i in np.arange(0, 256)]).astype("uint8")
        normalized = cv2.LUT(normalized, lookup_table)
        
        return normalized
    
    def normalize(self, img: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """Normalize an image using adaptive color normalization.
        
        Args:
            img: Input image as either:
                - numpy array of shape (H, W, 3) with dtype uint8
                - torch.Tensor of shape (C, H, W) or (B, C, H, W) with values in [0, 1]
            
        Returns:
            Normalized image in the same format as input
        """
        # Handle tensor input
        if isinstance(img, torch.Tensor):
            # Convert tensor to numpy for normalization
            if img.dim() == 4:  # Batch of images
                # Convert from (B, C, H, W) to (B, H, W, C)
                img_np = img.permute(0, 2, 3, 1).cpu().numpy()
                # Scale from [0, 1] to [0, 255]
                img_np = (img_np * 255).astype(np.uint8)
                # Normalize each image in the batch
                normalized = np.stack([self.normalize(img) for img in img_np])
                # Convert back to tensor format
                normalized = torch.from_numpy(normalized).float() / 255.0
                normalized = normalized.permute(0, 3, 1, 2)  # Back to (B, C, H, W)
                return normalized.to(img.device)
            else:  # Single image
                # Convert from (C, H, W) to (H, W, C)
                img_np = img.permute(1, 2, 0).cpu().numpy()
                # Scale from [0, 1] to [0, 255]
                img_np = (img_np * 255).astype(np.uint8)
                # Normalize
                normalized = self.normalize(img_np)
                # Convert back to tensor format
                normalized = torch.from_numpy(normalized).float() / 255.0
                normalized = normalized.permute(2, 0, 1)  # Back to (C, H, W)
                return normalized.to(img.device)
        
        # Handle numpy array input
        if not isinstance(img, np.ndarray):
            raise AttributeError("Input must be a numpy array or torch.Tensor")
        if img.size == 0:
            raise IndexError("Input array is empty")
        if len(img.shape) != 3 or img.shape[2] != 3:
            raise IndexError("Input must be a 3-channel image")
        if img.dtype != np.uint8:
            raise TypeError("Input must be uint8")
        
        # Try GPU first if enabled
        if self.use_gpu:
            try:
                return self._normalize_gpu(img)
            except Exception as e:
                logger.warning(f"GPU normalization failed, falling back to CPU: {str(e)}")
                self.use_gpu = False
                torch.cuda.empty_cache()
        
        # Use CPU (either by choice or after GPU failure)
        return self._normalize_cpu(img)

def create_normalizer(method: str = 'adaptive', use_gpu: bool = True) -> ColorNormalizer:
    """Create a color normalizer.
    
    Args:
        method: Normalization method ('dummy' or 'adaptive')
        use_gpu: Whether to use GPU acceleration (only for adaptive method).
                Defaults to True, will fall back to CPU if GPU is unavailable.
        
    Returns:
        ColorNormalizer instance
    """
    if method == 'dummy':
        return DummyNormalizer()
    elif method == 'adaptive':
        # Always try GPU first if requested, will fall back to CPU if unavailable
        return AdaptiveNormalizer(use_gpu=use_gpu and torch.cuda.is_available())
    else:
        raise ValueError(f"Unknown normalization method: {method}")

# Remove the alias - ColorNormalizer should remain an abstract base class
# ColorNormalizer = AdaptiveNormalizer  # This line should be removed 