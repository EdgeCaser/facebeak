class AdaptiveNormalizer:
    """Adaptive color normalization for crow images."""
    
    def __init__(self, target_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
                 target_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)):
        """Initialize normalizer with target statistics.
        
        Args:
            target_mean: Target mean RGB values (default: ImageNet stats)
            target_std: Target standard deviation RGB values (default: ImageNet stats)
        """
        self.target_mean = np.array(target_mean, dtype=np.float32)
        self.target_std = np.array(target_std, dtype=np.float32)
        self._validate_stats()
        
    def _validate_stats(self) -> None:
        """Validate target statistics."""
        if not (0 <= self.target_mean.min() <= 1 and 0 <= self.target_mean.max() <= 1):
            raise ValueError("Target mean values must be in range [0, 1]")
        if not (0 < self.target_std.min() <= 1 and 0 < self.target_std.max() <= 1):
            raise ValueError("Target std values must be in range (0, 1]")
            
    def normalize(self, image: np.ndarray) -> np.ndarray:
        """Normalize image to target statistics.
        
        Args:
            image: Input image array of shape (H, W, 3) with values in [0, 255]
            
        Returns:
            Normalized image array of shape (H, W, 3) with values in [0, 1]
            
        Raises:
            ValueError: If image shape or type is invalid
        """
        if not isinstance(image, np.ndarray):
            raise ValueError(f"Expected numpy array, got {type(image)}")
            
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected image shape (H, W, 3), got {image.shape}")
            
        # Convert to float32 if needed
        if image.dtype != np.float32:
            image = image.astype(np.float32)
            
        # Scale to [0, 1] if needed
        if image.max() > 1.0:
            image = image / 255.0
            
        # Compute current statistics
        current_mean = np.mean(image, axis=(0, 1))
        current_std = np.std(image, axis=(0, 1))
        
        # Avoid division by zero
        current_std = np.maximum(current_std, 1e-6)
        
        # Normalize to target statistics
        normalized = (image - current_mean) / current_std
        normalized = normalized * self.target_std + self.target_mean
        
        # Clip to valid range
        normalized = np.clip(normalized, 0.0, 1.0)
        
        return normalized

class EnhancedTracker:
    """Enhanced crow tracking with adaptive normalization and improved detection."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize enhanced tracker.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.tracker = None
        self.normalizer = AdaptiveNormalizer()
        self._initialize_tracker()
        
    def _initialize_tracker(self) -> None:
        """Initialize the underlying tracker."""
        tracker_type = self.config.get('tracker_type', 'CSRT')
        if tracker_type == 'CSRT':
            self.tracker = cv2.TrackerCSRT_create()
        elif tracker_type == 'KCF':
            self.tracker = cv2.TrackerKCF_create()
        else:
            raise ValueError(f"Unsupported tracker type: {tracker_type}")
            
    def start_tracking(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> bool:
        """Start tracking a crow in the given frame.
        
        Args:
            frame: Input frame as numpy array
            bbox: Initial bounding box (x, y, w, h)
            
        Returns:
            bool: True if tracking started successfully
        """
        if not isinstance(frame, np.ndarray):
            raise ValueError("Frame must be a numpy array")
            
        # Convert frame to RGB if needed
        if frame.shape[2] == 3 and frame.dtype == np.uint8:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
        # Normalize frame
        normalized_frame = self.normalizer.normalize(frame)
        
        # Convert back to uint8 for OpenCV
        normalized_frame = (normalized_frame * 255).astype(np.uint8)
        
        return self.tracker.init(normalized_frame, bbox)
        
    def update(self, frame: np.ndarray) -> Tuple[bool, Tuple[int, int, int, int]]:
        """Update tracking with new frame.
        
        Args:
            frame: New frame as numpy array
            
        Returns:
            Tuple of (success, bbox) where bbox is (x, y, w, h)
        """
        if not isinstance(frame, np.ndarray):
            raise ValueError("Frame must be a numpy array")
            
        # Convert frame to RGB if needed
        if frame.shape[2] == 3 and frame.dtype == np.uint8:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
        # Normalize frame
        normalized_frame = self.normalizer.normalize(frame)
        
        # Convert back to uint8 for OpenCV
        normalized_frame = (normalized_frame * 255).astype(np.uint8)
        
        return self.tracker.update(normalized_frame) 