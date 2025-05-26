import pytest
import numpy as np
import cv2
# Defer torch import to allow basic tests to run if torch is not installed (due to space constraints)
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from utilities.color_normalization import create_normalizer, AdaptiveNormalizer, ColorNormalizer

def create_test_image(size=(224, 224), color=(100, 100, 100), noise_level=20):
    """Create a test image with controlled color and noise."""
    img = np.ones((size[0], size[1], 3), dtype=np.uint8)
    img = img * np.array(color, dtype=np.uint8)
    noise = np.random.normal(0, noise_level, img.shape).astype(np.int16)
    img = np.clip(img + noise, 0, 255).astype(np.uint8)
    return img

def create_gradient_image(size=(224, 224)):
    """Create a test image with a gradient pattern."""
    x = np.linspace(0, 255, size[1])
    y = np.linspace(0, 255, size[0])
    X, Y = np.meshgrid(x, y)
    img = np.stack([X, Y, (X + Y) / 2], axis=2).astype(np.uint8)
    return img

def test_dummy_normalizer():
    """Test the dummy normalizer returns the input image unchanged."""
    normalizer = create_normalizer(method='dummy')
    test_img = create_test_image()
    
    # Test basic functionality
    normalized = normalizer.normalize(test_img)
    assert isinstance(normalized, np.ndarray)
    assert normalized.shape == test_img.shape
    assert normalized.dtype == test_img.dtype
    assert np.array_equal(normalized, test_img)
    
    # Test with different image sizes
    sizes = [(100, 100), (224, 224), (512, 512)]
    for size in sizes:
        img = create_test_image(size=size)
        normalized = normalizer.normalize(img)
        assert normalized.shape == size + (3,)
    
    # Test with different color ranges
    colors = [(0, 0, 0), (128, 128, 128), (255, 255, 255)]
    for color in colors:
        img = create_test_image(color=color)
        normalized = normalizer.normalize(img)
        assert np.array_equal(normalized, img)

def test_normalizer_invalid_input():
    """Test normalizer handles invalid inputs gracefully."""
    normalizer = create_normalizer(method='dummy')
    
    # Test with None
    with pytest.raises(AttributeError):
        normalizer.normalize(None)
    
    # Test with empty array
    with pytest.raises(IndexError):
        normalizer.normalize(np.array([]))
    
    # Test with wrong dimensions
    with pytest.raises(IndexError):
        normalizer.normalize(np.zeros((224, 224)))  # Missing color channel
    
    # Test with wrong data type
    with pytest.raises(TypeError):
        normalizer.normalize(np.zeros((224, 224, 3), dtype=np.float32))

def test_normalizer_methods():
    """Test different normalization methods."""
    methods = ['dummy', 'adaptive']  # Add more methods as they are implemented
    
    for method in methods:
        if method == 'adaptive' and not TORCH_AVAILABLE:
            pytest.skip("Skipping adaptive normalizer tests as torch is not available.")
        try:
            normalizer = create_normalizer(method=method)
            test_img = create_test_image()
            
            # Test basic functionality
            normalized = normalizer.normalize(test_img)
            assert isinstance(normalized, np.ndarray)
            assert normalized.shape == test_img.shape
            assert normalized.dtype == np.uint8
            
            # Test value range
            assert normalized.min() >= 0
            assert normalized.max() <= 255
            
            # Test that normalization preserves image structure
            if method != 'dummy':
                # For non-dummy methods, check that the image structure is preserved
                # by comparing gradients
                orig_grad = cv2.Laplacian(test_img, cv2.CV_64F)
                norm_grad = cv2.Laplacian(normalized, cv2.CV_64F)
                assert np.corrcoef(orig_grad.flatten(), norm_grad.flatten())[0, 1] > 0.5
                
        except NotImplementedError:
            # Skip if method is not implemented yet
            pytest.skip(f"Normalization method '{method}' not implemented")

def test_normalizer_gpu():
    """Test GPU acceleration if available."""
    # Create a test image
    img = create_test_image()
    
    # Test with GPU enabled (should use GPU if available)
    normalizer = create_normalizer(method='adaptive', use_gpu=True) # This will try to use torch
    if TORCH_AVAILABLE:
        if torch.cuda.is_available():
            assert normalizer.use_gpu, "GPU should be enabled when available"
            # Test that GPU normalization works
            normalized = normalizer.normalize(img)
            assert isinstance(normalized, np.ndarray)
            assert normalized.shape == img.shape
            assert normalized.dtype == np.uint8
            assert np.all(normalized >= 0) and np.all(normalized <= 255)
        else:
            assert not normalizer.use_gpu, "GPU should be disabled when unavailable"
    else:
        pytest.skip("Skipping GPU tests as torch is not available.")
    
    # Test with GPU explicitly disabled
    normalizer_no_gpu = create_normalizer(method='adaptive', use_gpu=False) # This will also try to use torch for torch.cuda.is_available
    if TORCH_AVAILABLE: # normalizer_no_gpu creation might fail if torch not present
        assert not normalizer_no_gpu.use_gpu, "GPU should be disabled when explicitly requested"
        normalized = normalizer_no_gpu.normalize(img)
        assert isinstance(normalized, np.ndarray)
        assert normalized.shape == img.shape
        assert normalized.dtype == np.uint8
        assert np.all(normalized >= 0) and np.all(normalized <= 255)
    elif not TORCH_AVAILABLE and method_is_adaptive_in_create_normalizer_uses_torch():
        # If create_normalizer itself fails for 'adaptive' without torch, this part is effectively skipped.
        # This is a complex interaction. The goal is to let non-torch tests pass.
        pass


# Helper to check if create_normalizer for 'adaptive' uses torch directly
# This is tricky because the code for create_normalizer itself uses torch.cuda.is_available()
# For simplicity, we assume tests below this point requiring 'adaptive' will be skipped if TORCH_AVAILABLE is false.

def method_is_adaptive_in_create_normalizer_uses_torch():
    # This is a placeholder for the complex check.
    # In create_normalizer: AdaptiveNormalizer(use_gpu=use_gpu and torch.cuda.is_available())
    # In AdaptiveNormalizer.__init__: self.use_gpu = use_gpu and torch.cuda.is_available()
    # So yes, it uses torch.
    return True


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not available")
def test_adaptive_normalizer_lab():
    """Test LAB color space conversion in adaptive normalizer."""
    normalizer = create_normalizer(method='adaptive')
    test_img = create_test_image()
    
    # Test LAB conversion
    normalized = normalizer.normalize(test_img)
    
    # Convert both images to LAB
    lab_orig = cv2.cvtColor(test_img, cv2.COLOR_BGR2LAB)
    lab_norm = cv2.cvtColor(normalized, cv2.COLOR_BGR2LAB)
    
    # Split channels
    l_orig, a_orig, b_orig = cv2.split(lab_orig)
    l_norm, a_norm, b_norm = cv2.split(lab_norm)
    
    # L channel should be different (due to CLAHE)
    assert not np.array_equal(l_orig, l_norm), "L channel should be modified by CLAHE"
    
    # a and b channels will be affected by gamma correction
    # but should still maintain their relative relationships
    a_corr = np.corrcoef(a_orig.flatten(), a_norm.flatten())[0, 1]
    b_corr = np.corrcoef(b_orig.flatten(), b_norm.flatten())[0, 1]
    assert a_corr > 0.9, "a channel should maintain relative relationships"
    assert b_corr > 0.9, "b channel should maintain relative relationships"
    
    # Check that the overall color relationships are preserved
    # by comparing the ratios between channels
    orig_ratio = a_orig.astype(float) / (b_orig + 1e-6)  # Avoid division by zero
    norm_ratio = a_norm.astype(float) / (b_norm + 1e-6)
    ratio_corr = np.corrcoef(orig_ratio.flatten(), norm_ratio.flatten())[0, 1]
    assert ratio_corr > 0.9, "Color relationships should be preserved"

@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not available")
def test_adaptive_normalizer_clahe():
    """Test CLAHE effects in adaptive normalizer."""
    normalizer = create_normalizer(method='adaptive')
    
    # Create a low contrast image
    img = create_test_image(color=(50, 50, 50))
    img = cv2.GaussianBlur(img, (5, 5), 0)  # Reduce contrast further
    
    normalized = normalizer.normalize(img)
    
    # Check that contrast is improved
    orig_std = np.std(img)
    norm_std = np.std(normalized)
    assert norm_std > orig_std, "Normalization should increase contrast"
    
    # Check that local contrast is improved
    orig_grad = cv2.Laplacian(img, cv2.CV_64F)
    norm_grad = cv2.Laplacian(normalized, cv2.CV_64F)
    assert np.std(norm_grad) > np.std(orig_grad), "Local contrast should be improved"

@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not available")
def test_adaptive_normalizer_gamma():
    """Test gamma correction in adaptive normalizer."""
    # Create a gradient image from dark to bright
    img = create_gradient_image()
    
    # Create normalizer with gamma correction
    normalizer = create_normalizer(method='adaptive', use_gpu=True)
    
    # Normalize the image
    normalized = normalizer.normalize(img)
    
    # Calculate mean brightness in dark and bright regions
    dark_region = img[:, :img.shape[1]//2]  # Left half (darker)
    bright_region = img[:, img.shape[1]//2:]  # Right half (brighter)
    dark_normalized = normalized[:, :normalized.shape[1]//2]
    bright_normalized = normalized[:, normalized.shape[1]//2:]
    
    # Calculate improvement (increase in brightness)
    dark_improvement = np.mean(dark_normalized) - np.mean(dark_region)
    bright_improvement = np.mean(bright_normalized) - np.mean(bright_region)
    
    # For gamma correction, we expect:
    # 1. Dark regions should be brightened by a meaningful amount
    # 2. Dark regions should be affected more than bright regions
    assert dark_improvement > 2.0, f"Dark regions should be brightened by at least 2.0, got {dark_improvement}"
    assert dark_improvement > bright_improvement, "Dark regions should be affected more than bright regions"

@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not available")
def test_adaptive_normalizer_gpu_fallback():
    """Test GPU fallback behavior."""
    if not torch.cuda.is_available(): # This check itself uses torch.
        pytest.skip("GPU not available for testing fallback")
    
    # Create a test image
    img = create_test_image()
    
    # Create normalizer with GPU enabled
    normalizer = create_normalizer(method='adaptive', use_gpu=True)
    assert normalizer.use_gpu, "GPU should be enabled"
    
    # Simulate GPU failure by forcing an error in GPU normalization
    original_normalize_gpu = normalizer._normalize_gpu
    try:
        def mock_normalize_gpu(img):
            raise RuntimeError("Simulated GPU failure")
        normalizer._normalize_gpu = mock_normalize_gpu
        
        # Should fall back to CPU
        normalized = normalizer.normalize(img)
        assert not normalizer.use_gpu, "Should have fallen back to CPU"
        assert isinstance(normalized, np.ndarray)
        assert normalized.shape == img.shape
        assert normalized.dtype == np.uint8
        assert np.all(normalized >= 0) and np.all(normalized <= 255)
    finally:
        # Restore original GPU normalization method
        normalizer._normalize_gpu = original_normalize_gpu

def test_normalizer_gpu_memory():
    """Test GPU memory management."""
    if not torch.cuda.is_available():
        pytest.skip("GPU not available for memory testing")
    
    # Create normalizer with GPU enabled
    normalizer = create_normalizer(method='adaptive', use_gpu=True)
    assert normalizer.use_gpu, "GPU should be enabled"
    
    # Create a large test image
    img = create_test_image(size=(1024, 1024))
    
    # Process multiple images to test memory management
    for _ in range(5):
        normalized = normalizer.normalize(img)
        assert isinstance(normalized, np.ndarray)
        assert normalized.shape == img.shape
        assert normalized.dtype == np.uint8
        
        # Check that GPU memory is being managed
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            memory_allocated = torch.cuda.memory_allocated() / (1024 ** 2)  # MB
            assert memory_allocated < 1000, f"GPU memory usage too high: {memory_allocated:.1f}MB"

def test_adaptive_normalizer_edge_cases():
    """Test adaptive normalizer with edge cases."""
    normalizer = create_normalizer(method='adaptive')
    
    # Test with very small image
    small_img = create_test_image(size=(10, 10))
    normalized = normalizer.normalize(small_img)
    assert normalized.shape == small_img.shape
    
    # Test with very bright image
    bright_img = create_test_image(color=(250, 250, 250))
    normalized = normalizer.normalize(bright_img)
    assert normalized.max() <= 255
    assert normalized.min() >= 0
    
    # Test with very dark image
    dark_img = create_test_image(color=(5, 5, 5))
    normalized = normalizer.normalize(dark_img)
    assert normalized.max() <= 255
    assert normalized.min() >= 0
    
    # Test with single color image
    single_color = create_test_image(color=(100, 100, 100), noise_level=0)
    normalized = normalizer.normalize(single_color)
    assert normalized.max() <= 255
    assert normalized.min() >= 0

# test_color_normalizer_base_class is fine as it does not use torch.

@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not available")
def test_adaptive_normalizer_gpu_init_error():
    """Test GPU initialization error handling."""
    if not torch.cuda.is_available(): # This check itself uses torch.
        pytest.skip("GPU not available for testing initialization error")
    
    # Mock torch.cuda.is_available to simulate GPU availability
    original_is_available = torch.cuda.is_available
    try:
        def mock_is_available():
            return True
        torch.cuda.is_available = mock_is_available
        
        # Mock torch.zeros to simulate GPU initialization failure
        original_zeros = torch.zeros
        def mock_zeros(*args, **kwargs):
            raise RuntimeError("Simulated GPU initialization failure")
        torch.zeros = mock_zeros
        
        # Should fall back to CPU
        normalizer = create_normalizer(method='adaptive', use_gpu=True)
        assert not normalizer.use_gpu, "Should have fallen back to CPU after GPU init failure"
        
        # Test that it still works on CPU
        test_img = create_test_image()
        normalized = normalizer.normalize(test_img)
        assert isinstance(normalized, np.ndarray)
        assert normalized.shape == test_img.shape
        assert normalized.dtype == np.uint8
    finally:
        # Restore original functions
        torch.cuda.is_available = original_is_available
        torch.zeros = original_zeros

@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not available")
def test_adaptive_normalizer_gpu_normalization_error():
    """Test GPU normalization error handling."""
    if not torch.cuda.is_available(): # This check itself uses torch.
        pytest.skip("GPU not available for testing normalization error")
    
    # Create normalizer with GPU enabled
    normalizer = create_normalizer(method='adaptive', use_gpu=True)
    assert normalizer.use_gpu, "GPU should be enabled"
    
    # Mock _normalize_gpu to simulate different types of errors
    original_normalize_gpu = normalizer._normalize_gpu
    test_img = create_test_image()
    
    try:
        # Test CUDA out of memory error
        def mock_normalize_gpu_oom(img):
            raise torch.cuda.OutOfMemoryError("Simulated CUDA OOM")
        normalizer._normalize_gpu = mock_normalize_gpu_oom
        normalized = normalizer.normalize(test_img)
        assert not normalizer.use_gpu, "Should have fallen back to CPU after OOM"
        assert isinstance(normalized, np.ndarray)
        
        # Test CUDA runtime error
        def mock_normalize_gpu_runtime(img):
            raise RuntimeError("Simulated CUDA runtime error")
        normalizer._normalize_gpu = mock_normalize_gpu_runtime
        normalized = normalizer.normalize(test_img)
        assert not normalizer.use_gpu, "Should have fallen back to CPU after runtime error"
        assert isinstance(normalized, np.ndarray)
        
        # Test general exception
        def mock_normalize_gpu_general(img):
            raise Exception("Simulated general error")
        normalizer._normalize_gpu = mock_normalize_gpu_general
        normalized = normalizer.normalize(test_img)
        assert not normalizer.use_gpu, "Should have fallen back to CPU after general error"
        assert isinstance(normalized, np.ndarray)
    finally:
        # Restore original GPU normalization method
        normalizer._normalize_gpu = original_normalize_gpu

def test_normalizer_unknown_method():
    """Test error handling for unknown normalization method."""
    with pytest.raises(ValueError, match="Unknown normalization method"):
        create_normalizer(method='unknown_method')

# Add skipif for other tests that might use adaptive normalizer implicitly
@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not available")
def test_normalizer_gpu_memory():
    """Test GPU memory management."""
    if not torch.cuda.is_available(): # This check itself uses torch.
        pytest.skip("GPU not available for memory testing")
    
    # Create normalizer with GPU enabled
    normalizer = create_normalizer(method='adaptive', use_gpu=True)
    assert normalizer.use_gpu, "GPU should be enabled"
    
    # Create a large test image
    img = create_test_image(size=(1024, 1024))
    
    # Process multiple images to test memory management
    for _ in range(5):
        normalized = normalizer.normalize(img)
        assert isinstance(normalized, np.ndarray)
        assert normalized.shape == img.shape
        assert normalized.dtype == np.uint8
        
        # Check that GPU memory is being managed
        if torch.cuda.is_available(): # This check itself uses torch.
            torch.cuda.empty_cache()
            memory_allocated = torch.cuda.memory_allocated() / (1024 ** 2)  # MB
            assert memory_allocated < 1000, f"GPU memory usage too high: {memory_allocated:.1f}MB"

@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not available")
def test_adaptive_normalizer_edge_cases():
    """Test adaptive normalizer with edge cases."""
    normalizer = create_normalizer(method='adaptive')
    
    # Test with very small image
    small_img = create_test_image(size=(10, 10))
    normalized = normalizer.normalize(small_img)
    assert normalized.shape == small_img.shape
    
    # Test with very bright image
    bright_img = create_test_image(color=(250, 250, 250))
    normalized = normalizer.normalize(bright_img)
    assert normalized.max() <= 255
    assert normalized.min() >= 0
    
    # Test with very dark image
    dark_img = create_test_image(color=(5, 5, 5))
    normalized = normalizer.normalize(dark_img)
    assert normalized.max() <= 255
    assert normalized.min() >= 0
    
    # Test with single color image
    single_color = create_test_image(color=(100, 100, 100), noise_level=0)
    normalized = normalizer.normalize(single_color)