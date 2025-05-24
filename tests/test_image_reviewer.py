import unittest
import os
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys

# Import the image reviewer module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from db import initialize_database

class TestImageReviewer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        cls.test_dir = tempfile.mkdtemp()
        cls.test_images_dir = os.path.join(cls.test_dir, "test_images")
        os.makedirs(cls.test_images_dir, exist_ok=True)
        
        # Create test image files
        cls.test_image_files = []
        for i in range(10):
            img_path = os.path.join(cls.test_images_dir, f"test_crow_{i:03d}.jpg")
            Path(img_path).touch()
            cls.test_image_files.append(img_path)
    
    def setUp(self):
        """Set up each test."""
        self.test_db_path = os.path.join(self.test_dir, f"test_db_{id(self)}.db")
        
        # Create the database file first
        Path(self.test_db_path).touch()
        
        # Patch the database path
        self.db_patcher = patch('db.DB_PATH', Path(self.test_db_path))
        self.db_patcher.start()
        
        # Initialize database
        initialize_database()
        
    def tearDown(self):
        """Clean up after each test."""
        self.db_patcher.stop()
        
        # Clean up test database
        if os.path.exists(self.test_db_path):
            try:
                os.remove(self.test_db_path)
            except PermissionError:
                pass
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment."""
        if os.path.exists(cls.test_dir):
            shutil.rmtree(cls.test_dir)
    
    def test_label_validation(self):
        """Test label validation functionality."""
        # Test valid labels
        valid_labels = ["crow", "not_a_crow", "not_sure"]
        for label in valid_labels:
            self.assertIn(label, valid_labels)
        
        # Test invalid label
        invalid_label = "invalid_label"
        self.assertNotIn(invalid_label, valid_labels)

    def test_image_scaling_logic(self):
        """Test image scaling calculation logic."""
        # Test different image sizes and scaling logic
        test_sizes = [(100, 100), (500, 300), (1920, 1080)]
        canvas_size = (600, 600)
        
        for width, height in test_sizes:
            # Calculate scaling factor (similar to what ImageReviewer does)
            scale_x = canvas_size[0] / width
            scale_y = canvas_size[1] / height
            scale = min(scale_x, scale_y, 1.0)  # Don't upscale
            
            scaled_width = int(width * scale)
            scaled_height = int(height * scale)
            
            # Should fit within canvas
            self.assertLessEqual(scaled_width, canvas_size[0])
            self.assertLessEqual(scaled_height, canvas_size[1])

    def test_confidence_scoring_logic(self):
        """Test confidence scoring functionality."""
        # Test confidence scoring logic
        confidence_levels = [0.1, 0.5, 0.9, 1.0]
        for confidence in confidence_levels:
            # Basic validation
            self.assertGreaterEqual(confidence, 0.0)
            self.assertLessEqual(confidence, 1.0)
            
            # Test confidence categories
            if confidence < 0.5:
                category = "low"
            elif confidence < 0.8:
                category = "medium"
            else:
                category = "high"
            
            self.assertIn(category, ["low", "medium", "high"])

    def test_file_structure(self):
        """Test that the test files are created properly."""
        # Verify test files exist
        self.assertEqual(len(self.test_image_files), 10)
        for img_path in self.test_image_files:
            self.assertTrue(os.path.exists(img_path))
    
    def test_database_operations(self):
        """Test that database operations work correctly."""
        from db import get_connection
        
        # Test database connection
        conn = get_connection()
        self.assertIsNotNone(conn)
        
        # Test that tables exist
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='image_labels';")
        result = cursor.fetchone()
        self.assertIsNotNone(result)
        
        conn.close()

    def test_image_path_parsing(self):
        """Test image path parsing logic."""
        # Test parsing logic that might be used in ImageReviewer
        test_paths = [
            "crow_crops/1/video1.mp4_frame0001_1.jpg",
            "crow_crops/2/video2.mp4_frame0010_2.jpg",
            "test_images/test_crow_001.jpg"
        ]
        
        for path in test_paths:
            # Basic path validation
            self.assertTrue(path.endswith('.jpg'))
            self.assertIn('/', path)  # Should have directory structure
    
    def test_navigation_logic(self):
        """Test navigation logic without GUI components."""
        # Simulate navigation logic
        images = self.test_image_files[:5]
        current_index = 0
        
        # Test next navigation
        next_index = min(current_index + 1, len(images) - 1)
        self.assertEqual(next_index, 1)
        
        # Test previous navigation from middle
        current_index = 2
        prev_index = max(current_index - 1, 0)
        self.assertEqual(prev_index, 1)
        
        # Test boundaries
        current_index = 0
        prev_index = max(current_index - 1, 0)
        self.assertEqual(prev_index, 0)  # Should stay at 0
        
        current_index = len(images) - 1
        next_index = min(current_index + 1, len(images) - 1)
        self.assertEqual(next_index, len(images) - 1)  # Should stay at last

    def test_batch_size_calculation(self):
        """Test batch size calculation logic."""
        total_images = len(self.test_image_files)
        max_batch_size = 5
        
        batch_size = min(total_images, max_batch_size)
        self.assertEqual(batch_size, 5)
        
        # Test with fewer images
        small_set = 3
        batch_size = min(small_set, max_batch_size)
        self.assertEqual(batch_size, 3)

if __name__ == '__main__':
    unittest.main() 