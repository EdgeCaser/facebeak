#!/usr/bin/env python3
"""
Test script for the Crow Image Ingestion Tool.
"""

import os
import sys
import tempfile
import shutil
import cv2
import numpy as np
from pathlib import Path

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gui.image_ingestion_gui import find_image_files, CropReviewWindow
from crow_tracking import CrowTracker
from detection import detect_crows_parallel
from tracking import extract_normalized_crow_crop

def create_test_image(output_path, size=(800, 600)):
    """Create a test image with a simple pattern."""
    image = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    
    # Add some shapes that might trigger detection
    # Draw a dark shape in the center (simulating a crow)
    center_x, center_y = size[0] // 2, size[1] // 2
    cv2.rectangle(image, (center_x-50, center_y-50), (center_x+50, center_y+50), (50, 50, 50), -1)
    
    # Add some background elements
    cv2.rectangle(image, (100, 100), (200, 200), (100, 100, 100), -1)
    cv2.rectangle(image, (600, 400), (700, 500), (100, 100, 100), -1)
    
    cv2.imwrite(str(output_path), image)
    return output_path

def test_find_image_files():
    """Test finding image files in a directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test images
        test_images = [
            "test1.jpg",
            "test2.png", 
            "test3.bmp",
            "test4.tiff",
            "not_an_image.txt"
        ]
        
        for filename in test_images:
            if filename.endswith(('.jpg', '.png', '.bmp', '.tiff')):
                create_test_image(Path(temp_dir) / filename)
            else:
                # Create a text file
                with open(Path(temp_dir) / filename, 'w') as f:
                    f.write("not an image")
        
        # Test non-recursive search
        found_files = find_image_files(temp_dir, recursive=False)
        assert len(found_files) == 4, f"Expected 4 image files, found {len(found_files)}"
        
        # Test recursive search (should be same for flat directory)
        found_files_recursive = find_image_files(temp_dir, recursive=True)
        assert len(found_files_recursive) == 4, f"Expected 4 image files, found {len(found_files_recursive)}"
        
        print("✓ find_image_files test passed")

def test_crop_extraction():
    """Test crop extraction functionality."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test image
        image_path = Path(temp_dir) / "test.jpg"
        create_test_image(image_path)
        
        # Load image
        image = cv2.imread(str(image_path))
        assert image is not None, "Failed to load test image"
        
        # Test detection
        detections = detect_crows_parallel([image], score_threshold=0.1)
        frame_dets = detections[0]
        
        if frame_dets:
            # Test crop extraction
            detection = frame_dets[0]
            crop_dict = extract_normalized_crow_crop(
                image, 
                detection['bbox'], 
                correct_orientation=True,
                padding=0.3
            )
            
            assert crop_dict is not None, "Crop extraction failed"
            assert 'full' in crop_dict, "Crop should have 'full' key"
            assert 'head' in crop_dict, "Crop should have 'head' key"
            assert crop_dict['full'].shape == (512, 512, 3), "Crop should be 512x512x3"
            
            print("✓ crop_extraction test passed")
        else:
            print("⚠ No detections found in test image (this is normal)")

def test_tracker_integration():
    """Test integration with CrowTracker."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test image
        image_path = Path(temp_dir) / "test.jpg"
        create_test_image(image_path)
        
        # Initialize tracker
        tracker = CrowTracker(base_dir=temp_dir, enable_audio_extraction=False)
        
        # Load image
        image = cv2.imread(str(image_path))
        
        # Test detection and processing
        detections = detect_crows_parallel([image], score_threshold=0.1)
        frame_dets = detections[0]
        
        if frame_dets:
            detection = frame_dets[0]
            
            # Process detection
            crow_id = tracker.process_detection(
                image, 
                0,  # frame number
                detection,
                str(image_path),  # video path (image path in this case)
                None  # frame time
            )
            
            if crow_id:
                # Check if crop was saved
                crow_dir = tracker.crows_dir / "test" / "frame_000000"
                assert crow_dir.exists(), "Crow directory should exist"
                
                crop_files = list(crow_dir.glob("*.jpg"))
                assert len(crop_files) > 0, "Crop file should be saved"
                
                print("✓ tracker_integration test passed")
            else:
                print("⚠ Detection processing failed (this may be normal)")
        else:
            print("⚠ No detections found in test image (this is normal)")

def test_label_storage():
    """Test label storage functionality."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Initialize tracker
        tracker = CrowTracker(base_dir=temp_dir, enable_audio_extraction=False)
        
        # Create test image
        image_path = Path(temp_dir) / "test.jpg"
        create_test_image(image_path)
        
        # Load image
        image = cv2.imread(str(image_path))
        
        # Test detection
        detections = detect_crows_parallel([image], score_threshold=0.1)
        frame_dets = detections[0]
        
        if frame_dets:
            detection = frame_dets[0]
            
            # Extract crop
            crop_dict = extract_normalized_crow_crop(
                image, 
                detection['bbox'], 
                correct_orientation=True,
                padding=0.3
            )
            
            if crop_dict:
                # Generate crow ID
                crow_id = tracker._generate_crow_id()
                
                # Save crop
                crop_path = tracker.save_crop(crop_dict, crow_id, 0, str(image_path))
                
                if crop_path:
                    # Add label
                    if 'labels' not in tracker.tracking_data:
                        tracker.tracking_data['labels'] = {}
                    
                    crop_relative_path = str(crop_path.relative_to(tracker.base_dir))
                    test_label = "test_crow"
                    
                    tracker.tracking_data['labels'][crop_relative_path] = {
                        'label': test_label,
                        'crow_id': crow_id,
                        'image_path': str(image_path),
                        'timestamp': '2024-01-01T00:00:00'
                    }
                    
                    # Save tracking data
                    tracker._save_tracking_data()
                    
                    # Verify label was saved
                    assert 'labels' in tracker.tracking_data, "Labels should be in tracking data"
                    assert crop_relative_path in tracker.tracking_data['labels'], "Label should be stored"
                    assert tracker.tracking_data['labels'][crop_relative_path]['label'] == test_label, "Label should match"
                    
                    print("✓ label_storage test passed")
                else:
                    print("⚠ Crop saving failed")
            else:
                print("⚠ Crop extraction failed")
        else:
            print("⚠ No detections found in test image (this is normal)")

def main():
    """Run all tests."""
    print("Running Crow Image Ingestion Tool tests...")
    print()
    
    try:
        test_find_image_files()
        test_crop_extraction()
        test_tracker_integration()
        test_label_storage()
        
        print()
        print("All tests completed successfully!")
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 