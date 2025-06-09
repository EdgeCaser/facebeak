import unittest
import numpy as np
import cv2
from multi_view import MultiViewExtractor, create_multi_view_extractor
import logging

# Configure logging for tests
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class TestMultiViewExtractor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures that are shared across all tests."""
        # Create a simple test image (100x100 white square with black border)
        cls.test_image = np.ones((100, 100, 3), dtype=np.uint8) * 255
        cv2.rectangle(cls.test_image, (0, 0), (99, 99), (0, 0, 0), 2)
        
        # Create a small test image
        cls.small_image = np.ones((50, 50, 3), dtype=np.uint8) * 255
        
        # Create a large test image
        cls.large_image = np.ones((2000, 2000, 3), dtype=np.uint8) * 255
    
    def setUp(self):
        """Set up test fixtures that are run before each test."""
        self.extractor = create_multi_view_extractor(
            rotation_angles=[-30, 30],
            zoom_factors=[1.2],
            min_size=100,
            max_size=1000
        )
    
    def test_initialization(self):
        """Test that the extractor initializes with correct parameters."""
        self.assertEqual(self.extractor.rotation_angles, [-30, 30])
        self.assertEqual(self.extractor.zoom_factors, [1.2])
        self.assertEqual(self.extractor.min_size, 100)
        self.assertEqual(self.extractor.max_size, 1000)
        self.assertEqual(self.extractor.interpolation, cv2.INTER_LINEAR)
    
    def test_extract_basic(self):
        """Test basic extraction with default parameters."""
        views = self.extractor.extract(self.test_image)
        
        # Should get original + 2 rotations + 1 zoom = 4 views
        self.assertEqual(len(views), 4)
        
        # Check that all views are valid images
        for view in views:
            self.assertIsInstance(view, np.ndarray)
            self.assertEqual(view.dtype, np.uint8)
            self.assertEqual(len(view.shape), 3)  # Should be BGR image
            self.assertEqual(view.shape[2], 3)  # Should have 3 channels
    
    def test_extract_small_image(self):
        """Test extraction with an image smaller than min_size."""
        views = self.extractor.extract(self.small_image)
        
        # Check that image was upscaled
        self.assertGreaterEqual(min(views[0].shape[:2]), self.extractor.min_size)
        
        # Should still get all views
        self.assertEqual(len(views), 4)
    
    def test_extract_large_image(self):
        """Test extraction with an image larger than max_size."""
        views = self.extractor.extract(self.large_image)
        
        # Check that image was downscaled
        self.assertLessEqual(max(views[0].shape[:2]), self.extractor.max_size)
        
        # Should still get all views
        self.assertEqual(len(views), 4)
    
    def test_extract_invalid_input(self):
        """Test extraction with invalid inputs."""
        # Test with None
        views = self.extractor.extract(None)
        self.assertEqual(len(views), 0)
        
        # Test with empty array
        views = self.extractor.extract(np.array([]))
        self.assertEqual(len(views), 0)
        
        # Test with wrong shape (2D array)
        views = self.extractor.extract(np.ones((100, 100)))
        self.assertEqual(len(views), 0)
    
    def test_rotation(self):
        """Test that rotations are applied correctly."""
        # Create an image with a clear orientation (e.g., a triangle)
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        pts = np.array([[50, 10], [10, 90], [90, 90]], np.int32)
        cv2.fillPoly(img, [pts], (255, 255, 255))
        
        views = self.extractor.extract(img)
        
        # Check that rotated views are different from original
        original = views[0]
        rotated_neg = views[1]  # -30 degrees
        rotated_pos = views[2]  # +30 degrees
        
        # Rotated views should be different from original
        self.assertFalse(np.array_equal(original, rotated_neg))
        self.assertFalse(np.array_equal(original, rotated_pos))
        self.assertFalse(np.array_equal(rotated_neg, rotated_pos))
    
    def test_zoom(self):
        """Test that zoom is applied correctly."""
        views = self.extractor.extract(self.test_image)
        
        # Get the zoomed view (last view)
        zoomed = views[-1]
        
        # Check that zoomed view is larger
        self.assertGreater(zoomed.shape[0], self.test_image.shape[0])
        self.assertGreater(zoomed.shape[1], self.test_image.shape[1])
    
    def test_custom_parameters(self):
        """Test extraction with custom parameters."""
        custom_extractor = create_multi_view_extractor(
            rotation_angles=[-45, 45],
            zoom_factors=[0.8, 1.5],
            min_size=200,
            max_size=500
        )
        
        views = custom_extractor.extract(self.test_image)
        
        # Should get original + 2 rotations + 2 zooms = 5 views
        self.assertEqual(len(views), 5)
        
        # Check that parameters were applied
        self.assertEqual(custom_extractor.rotation_angles, [-45, 45])
        self.assertEqual(custom_extractor.zoom_factors, [0.8, 1.5])
        self.assertEqual(custom_extractor.min_size, 200)
        self.assertEqual(custom_extractor.max_size, 500)
    
    def test_preserve_aspect_ratio(self):
        """Test that aspect ratio is preserved during resizing."""
        # Create a non-square image
        img = np.ones((200, 100, 3), dtype=np.uint8) * 255
        
        views = self.extractor.extract(img)
        
        # Check that aspect ratio is preserved in all views
        original_ratio = img.shape[1] / img.shape[0]
        for view in views:
            view_ratio = view.shape[1] / view.shape[0]
            self.assertAlmostEqual(view_ratio, original_ratio, places=2)

if __name__ == '__main__':
    unittest.main() 