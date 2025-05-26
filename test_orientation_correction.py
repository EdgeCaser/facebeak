#!/usr/bin/env python3
"""
Test script for crow orientation correction functionality.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from crow_orientation import correct_crow_crop_orientation, detect_crow_orientation

def create_test_crow():
    """Create a synthetic crow-like shape for testing."""
    # Create a 200x150 image (taller than wide)
    img = np.zeros((200, 150, 3), dtype=np.uint8)
    
    # Draw a simple crow-like shape
    # Head (circle at top)
    cv2.circle(img, (75, 40), 20, (50, 50, 50), -1)
    
    # Body (ellipse in middle)
    cv2.ellipse(img, (75, 100), (30, 50), 0, 0, 360, (40, 40, 40), -1)
    
    # Wings (ellipses on sides)
    cv2.ellipse(img, (50, 90), (15, 30), 20, 0, 360, (30, 30, 30), -1)
    cv2.ellipse(img, (100, 90), (15, 30), -20, 0, 360, (30, 30, 30), -1)
    
    # Legs (thin rectangles at bottom)
    cv2.rectangle(img, (70, 170), (75, 190), (60, 60, 60), -1)
    cv2.rectangle(img, (75, 170), (80, 190), (60, 60, 60), -1)
    
    return img

def test_orientation_detection():
    """Test orientation detection on rotated crow images."""
    print("Testing crow orientation detection...")
    
    # Create test crow
    original_crow = create_test_crow()
    
    # Create rotated versions
    rotations = [0, 90, 180, 270]
    test_images = []
    
    for rotation in rotations:
        if rotation == 0:
            rotated = original_crow.copy()
        elif rotation == 90:
            rotated = cv2.rotate(original_crow, cv2.ROTATE_90_CLOCKWISE)
        elif rotation == 180:
            rotated = cv2.rotate(original_crow, cv2.ROTATE_180)
        elif rotation == 270:
            rotated = cv2.rotate(original_crow, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        test_images.append((rotation, rotated))
    
    # Test detection on each rotation
    print("\nDetection Results:")
    print("Original -> Detected Correction")
    print("-" * 35)
    
    for orig_rotation, img in test_images:
        detected_rotation, detected_flip = detect_crow_orientation(img)
        corrected_img = correct_crow_crop_orientation(img)
        
        print(f"{orig_rotation:3d}°      -> {detected_rotation:3d}° rotation, flip={detected_flip}")
    
    # Create visualization
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle('Crow Orientation Correction Test', fontsize=16)
    
    for i, (orig_rotation, img) in enumerate(test_images):
        # Original (rotated) image
        axes[0, i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[0, i].set_title(f'Input: {orig_rotation}° rotation')
        axes[0, i].axis('off')
        
        # Corrected image
        corrected = correct_crow_crop_orientation(img)
        axes[1, i].imshow(cv2.cvtColor(corrected, cv2.COLOR_BGR2RGB))
        axes[1, i].set_title('Corrected')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig('orientation_correction_test.png', dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved as 'orientation_correction_test.png'")
    
    return True

def test_real_crow_crop():
    """Test with a real crow crop if available."""
    # This would be used with actual crow crops from the dataset
    print("\nTo test with real crow crops:")
    print("1. Extract some crops using the GUI")
    print("2. Load a crop image")
    print("3. Apply orientation correction")
    print("4. Compare results visually")

if __name__ == "__main__":
    print("Crow Orientation Correction Test")
    print("=" * 35)
    
    try:
        success = test_orientation_detection()
        if success:
            print("\n✅ Orientation correction test completed successfully!")
            print("Check the generated 'orientation_correction_test.png' file.")
        else:
            print("\n❌ Test failed")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    test_real_crow_crop() 