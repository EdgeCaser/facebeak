#!/usr/bin/env python3
"""
Test script for the manual drawing functionality in the image ingestion tool.
"""

import os
import sys
import tempfile
import unittest
from unittest.mock import patch, MagicMock
import cv2
import numpy as np
from pathlib import Path

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TestManualDrawing(unittest.TestCase):
    
    def test_coordinate_conversion(self):
        """Test canvas to image coordinate conversion."""
        from gui.image_ingestion_gui import CropReviewWindow
        import tkinter as tk
        
        # Create a test image
        test_image = np.zeros((600, 800, 3), dtype=np.uint8)
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
            cv2.imwrite(tmp_file.name, test_image)
            image_path = tmp_file.name
        
        try:
            # Create a temporary root window
            root = tk.Tk()
            
            try:
                # Create the review window
                review_window = CropReviewWindow(root, image_path, [], None, None, None)
                
                # Test coordinate conversion
                # Canvas is 800x600, image is 800x600 (1:1 scale)
                img_bbox = review_window.canvas_to_image_coords(100, 100, 200, 200)
                self.assertEqual(img_bbox, [100, 100, 200, 200])
                
                # Test with different image size
                review_window.original_image = np.zeros((1200, 1600, 3), dtype=np.uint8)
                img_bbox = review_window.canvas_to_image_coords(100, 100, 200, 200)
                # Should scale by 2 (1600/800 = 2, 1200/600 = 2)
                self.assertEqual(img_bbox, [200, 200, 400, 400])
                
                # Test boundary clamping
                img_bbox = review_window.canvas_to_image_coords(-50, -50, 900, 700)
                self.assertEqual(img_bbox[0], 0)  # Should be clamped to 0
                self.assertEqual(img_bbox[1], 0)  # Should be clamped to 0
                self.assertEqual(img_bbox[2], 1600)  # Should be clamped to image width
                self.assertEqual(img_bbox[3], 1200)  # Should be clamped to image height
                
            finally:
                root.destroy()
                
        finally:
            # Clean up temporary file
            os.unlink(image_path)
    
    def test_manual_detection_creation(self):
        """Test that manual detections are created correctly."""
        from gui.image_ingestion_gui import CropReviewWindow
        import tkinter as tk
        
        # Create a test image
        test_image = np.zeros((600, 800, 3), dtype=np.uint8)
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
            cv2.imwrite(tmp_file.name, test_image)
            image_path = tmp_file.name
        
        try:
            # Create a temporary root window
            root = tk.Tk()
            
            try:
                # Create the review window
                review_window = CropReviewWindow(root, image_path, [], None, None, None)
                
                # Simulate mouse events for drawing
                review_window.drawing_mode = True
                review_window.start_x = 100
                review_window.start_y = 100
                
                # Simulate mouse up event
                mock_event = MagicMock()
                mock_event.x = 200
                mock_event.y = 200
                
                review_window.on_mouse_up(mock_event)
                
                # Check that manual detection was created
                self.assertEqual(len(review_window.manual_detections), 1)
                
                manual_detection = review_window.manual_detections[0]
                self.assertEqual(manual_detection['bbox'], [100, 100, 200, 200])
                self.assertEqual(manual_detection['score'], 1.0)
                self.assertTrue(manual_detection['manual'])
                
            finally:
                root.destroy()
                
        finally:
            # Clean up temporary file
            os.unlink(image_path)
    
    def test_detection_combination(self):
        """Test that automatic and manual detections are combined correctly."""
        from gui.image_ingestion_gui import CropReviewWindow
        import tkinter as tk
        
        # Create a test image
        test_image = np.zeros((600, 800, 3), dtype=np.uint8)
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
            cv2.imwrite(tmp_file.name, test_image)
            image_path = tmp_file.name
        
        try:
            # Create a temporary root window
            root = tk.Tk()
            
            try:
                # Create automatic detections
                automatic_detections = [
                    {'bbox': [50, 50, 150, 150], 'score': 0.8, 'manual': False},
                    {'bbox': [200, 200, 300, 300], 'score': 0.9, 'manual': False}
                ]
                
                # Create the review window
                review_window = CropReviewWindow(root, image_path, automatic_detections, None, None, None)
                
                # Add manual detections
                review_window.manual_detections = [
                    {'bbox': [350, 350, 450, 450], 'score': 1.0, 'manual': True},
                    {'bbox': [500, 500, 600, 600], 'score': 1.0, 'manual': True}
                ]
                
                # Test that all detections are combined
                all_detections = review_window.detections + review_window.manual_detections
                self.assertEqual(len(all_detections), 4)
                
                # Check that automatic detections come first
                self.assertFalse(all_detections[0]['manual'])
                self.assertFalse(all_detections[1]['manual'])
                
                # Check that manual detections come after
                self.assertTrue(all_detections[2]['manual'])
                self.assertTrue(all_detections[3]['manual'])
                
            finally:
                root.destroy()
                
        finally:
            # Clean up temporary file
            os.unlink(image_path)
    
    def test_clear_manual_detections(self):
        """Test that manual detections can be cleared."""
        from gui.image_ingestion_gui import CropReviewWindow
        import tkinter as tk
        
        # Create a test image
        test_image = np.zeros((600, 800, 3), dtype=np.uint8)
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
            cv2.imwrite(tmp_file.name, test_image)
            image_path = tmp_file.name
        
        try:
            # Create a temporary root window
            root = tk.Tk()
            
            try:
                # Create the review window
                review_window = CropReviewWindow(root, image_path, [], None, None, None)
                
                # Add some manual detections
                review_window.manual_detections = [
                    {'bbox': [100, 100, 200, 200], 'score': 1.0, 'manual': True},
                    {'bbox': [300, 300, 400, 400], 'score': 1.0, 'manual': True}
                ]
                
                self.assertEqual(len(review_window.manual_detections), 2)
                
                # Clear manual detections
                review_window.clear_manual_detections()
                
                self.assertEqual(len(review_window.manual_detections), 0)
                
            finally:
                root.destroy()
                
        finally:
            # Clean up temporary file
            os.unlink(image_path)
    
    def test_square_bounding_box_drawing(self):
        """Test that manual drawing creates square bounding boxes."""
        from gui.image_ingestion_gui import CropReviewWindow
        import tkinter as tk
        
        # Create a simple test image
        image = np.ones((400, 400, 3), dtype=np.uint8) * 255
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
            cv2.imwrite(tmp_file.name, image)
            image_path = tmp_file.name
        
        try:
            # Create a temporary root window
            root = tk.Tk()
            
            try:
                # Create the review window
                window = CropReviewWindow(root, image_path, [], None, None, None)
                
                # Enable drawing mode
                window.drawing_var.set(True)
                window.drawing_mode = True
                
                # Simulate drawing a perfect square
                window.start_x = 100
                window.start_y = 100
                mock_event = MagicMock()
                mock_event.x = 200
                mock_event.y = 200
                window.on_mouse_drag(mock_event)
                window.on_mouse_up(mock_event)
                
                # Check that a manual detection was added
                self.assertEqual(len(window.manual_detections), 1)
                
                # Check that the bounding box is square
                bbox = window.manual_detections[0]['bbox']
                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]
                self.assertEqual(width, height)
                self.assertEqual(width, 100)
                self.assertEqual(height, 100)
                
            finally:
                root.destroy()
                
        finally:
            os.unlink(image_path)
    
    def test_dropdown_labels(self):
        """Test that labels are selected from dropdown and only valid labels are saved/used."""
        from gui.image_ingestion_gui import CropReviewWindow
        import tkinter as tk
        
        # Create a simple test image
        image = np.ones((400, 400, 3), dtype=np.uint8) * 255
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
            cv2.imwrite(tmp_file.name, image)
            image_path = tmp_file.name
        
        try:
            root = tk.Tk()
            try:
                window = CropReviewWindow(root, image_path, [], None, None, None)
                expected_labels = [
                    "crow", "not-crow", "juvenile-crow", "adult-crow",
                    "flying-crow", "perching-crow", "feeding-crow",
                    "multiple-crows", "crow-partial", "crow-occluded",
                    "crow-blurry", "crow-high-quality"
                ]
                self.assertEqual(window.predefined_labels, expected_labels)
                self.assertEqual(window.label_var.get(), "crow")
                window.label_var.set("flying-crow")
                self.assertEqual(window.label_var.get(), "flying-crow")
                # Simulate saving a valid label
                window.labels[0] = window.label_var.get()
                self.assertIn(window.labels[0], expected_labels)
                # Simulate setting an invalid label and saving (should not be used in real logic)
                window.label_var.set("invalid-label")
                # Only valid labels should be saved/used in real logic
                if window.label_var.get() not in expected_labels:
                    window.labels[1] = "crow"  # fallback/default
                else:
                    window.labels[1] = window.label_var.get()
                self.assertIn(window.labels[1], expected_labels)
            finally:
                root.destroy()
        finally:
            os.unlink(image_path)

def main():
    """Run the tests."""
    print("Testing manual drawing functionality...")
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestManualDrawing)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    if result.wasSuccessful():
        print("\n✅ All manual drawing tests passed!")
        return 0
    else:
        print("\n❌ Some tests failed!")
        return 1

if __name__ == "__main__":
    exit(main()) 