#!/usr/bin/env python3
"""
Verify that orientation correction is working on saved crop files.
"""

import cv2
import numpy as np
import os
import sys
from pathlib import Path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from crow_orientation import correct_crow_crop_orientation, detect_crow_orientation
import matplotlib.pyplot as plt

def verify_orientation_correction(crop_dir="crow_crops"):
    """
    Check a few saved crops to verify orientation correction is working.
    """
    crop_path = Path(crop_dir)
    if not crop_path.exists():
        print(f"Crop directory not found: {crop_dir}")
        return
    
    # Find some crop files
    crop_files = []
    for crow_dir in crop_path.glob("crows/crow_*"):
        if crow_dir.is_dir():
            for img_file in crow_dir.glob("*.jpg"):
                crop_files.append(img_file)
                if len(crop_files) >= 4:  # Just check first 4
                    break
            if len(crop_files) >= 4:
                break
    
    if not crop_files:
        print("No crop files found to verify")
        return
    
    print(f"Found {len(crop_files)} crop files to verify")
    
    # Check each crop
    fig, axes = plt.subplots(2, len(crop_files), figsize=(4*len(crop_files), 8))
    if len(crop_files) == 1:
        axes = axes.reshape(-1, 1)
    
    for i, crop_file in enumerate(crop_files):
        try:
            # Load the saved crop (this should already be orientation-corrected)
            saved_crop = cv2.imread(str(crop_file))
            if saved_crop is None:
                print(f"Could not load {crop_file}")
                continue
            
            # Convert BGR to RGB for matplotlib
            saved_crop_rgb = cv2.cvtColor(saved_crop, cv2.COLOR_BGR2RGB)
            
            # Apply orientation correction again to see if it changes
            corrected_crop = correct_crow_crop_orientation(saved_crop)
            corrected_crop_rgb = cv2.cvtColor(corrected_crop, cv2.COLOR_BGR2RGB)
            
            # Detect orientation of saved crop
            rotation, flip = detect_crow_orientation(saved_crop)
            
            # Show original saved crop
            axes[0, i].imshow(saved_crop_rgb)
            axes[0, i].set_title(f'Saved Crop\n{crop_file.name}')
            axes[0, i].axis('off')
            
            # Show re-corrected crop
            axes[1, i].imshow(corrected_crop_rgb)
            axes[1, i].set_title(f'Re-corrected\nDetected: {rotation}°, flip={flip}')
            axes[1, i].axis('off')
            
            # Check if they're the same (orientation correction already applied)
            diff = np.sum(np.abs(saved_crop_rgb.astype(float) - corrected_crop_rgb.astype(float)))
            if diff < 1000:  # Very small difference
                print(f"✅ {crop_file.name}: Already properly oriented")
            else:
                print(f"⚠️  {crop_file.name}: Orientation correction would change it (rotation={rotation}°, flip={flip})")
            
        except Exception as e:
            print(f"Error processing {crop_file}: {e}")
    
    plt.tight_layout()
    plt.savefig('orientation_verification.png', dpi=150, bbox_inches='tight')
    print(f"\nVerification plot saved as 'orientation_verification.png'")
    
    return True

if __name__ == "__main__":
    print("Verifying Orientation Correction on Saved Crops")
    print("=" * 45)
    
    verify_orientation_correction() 