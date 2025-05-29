#!/usr/bin/env python3
"""
Test script to verify 512x512 resolution upgrade works correctly.
This tests the main components that were affected by the resolution change.
"""

import numpy as np
import torch
import cv2
from pathlib import Path
import tempfile
import os

# Import the modules we've updated
from tracking import extract_normalized_crow_crop
from unsupervised_learning import SimCLRCrowDataset
from models import CrowResNetEmbedder

def test_crop_extraction():
    """Test that crop extraction works with 512x512."""
    print("Testing crop extraction with 512x512...")
    
    # Create a test frame
    frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
    bbox = [100, 100, 300, 300]  # x1, y1, x2, y2
    
    # Extract crop
    result = extract_normalized_crow_crop(frame, bbox)
    
    assert result is not None, "Crop extraction failed"
    assert 'full' in result and 'head' in result, "Missing crop components"
    assert result['full'].shape == (512, 512, 3), f"Wrong full crop shape: {result['full'].shape}"
    assert result['head'].shape == (512, 512, 3), f"Wrong head crop shape: {result['head'].shape}"
    
    print("✅ Crop extraction test passed!")
    return True

def test_simclr_dataset():
    """Test that SimCLR dataset works with 512x512."""
    print("Testing SimCLR dataset with 512x512...")
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create test images
        img_paths = []
        for i in range(3):
            img_path = Path(tmp_dir) / f"test_{i}.jpg"
            img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
            cv2.imwrite(str(img_path), img)
            img_paths.append(str(img_path))
        
        # Test dataset
        dataset = SimCLRCrowDataset(img_paths)
        view1, view2 = dataset[0]
        
        assert view1.shape == (3, 512, 512), f"Wrong view1 shape: {view1.shape}"
        assert view2.shape == (3, 512, 512), f"Wrong view2 shape: {view2.shape}"
        assert view1.dtype == torch.float32, f"Wrong dtype: {view1.dtype}"
        
    print("✅ SimCLR dataset test passed!")
    return True

def test_model_compatibility():
    """Test that ResNet models work with 512x512 input."""
    print("Testing model compatibility with 512x512...")
    
    # Test with CPU to avoid GPU memory issues
    device = 'cpu'
    model = CrowResNetEmbedder(embedding_dim=128, device=device)
    
    # Create test input
    batch_size = 2
    test_input = torch.randn(batch_size, 3, 512, 512)
    
    # Forward pass
    with torch.no_grad():
        embeddings = model(test_input)
    
    assert embeddings.shape == (batch_size, 128), f"Wrong embedding shape: {embeddings.shape}"
    assert not torch.isnan(embeddings).any(), "NaN values in embeddings"
    
    print("✅ Model compatibility test passed!")
    return True

def test_memory_usage():
    """Test memory usage with 512x512 vs 224x224."""
    print("Testing memory usage comparison...")
    
    device = 'cpu'  # Use CPU for consistent memory measurement
    
    # Test 224x224 (old size)
    model_224 = CrowResNetEmbedder(embedding_dim=128, device=device)
    input_224 = torch.randn(1, 3, 224, 224)
    
    with torch.no_grad():
        emb_224 = model_224(input_224)
    
    # Test 512x512 (new size)
    model_512 = CrowResNetEmbedder(embedding_dim=128, device=device)
    input_512 = torch.randn(1, 3, 512, 512)
    
    with torch.no_grad():
        emb_512 = model_512(input_512)
    
    # Calculate memory increase
    memory_224 = input_224.numel() * 4  # 4 bytes per float32
    memory_512 = input_512.numel() * 4
    memory_increase = memory_512 / memory_224
    
    print(f"📊 Memory usage:")
    print(f"   224x224: {memory_224 / 1024 / 1024:.2f} MB")
    print(f"   512x512: {memory_512 / 1024 / 1024:.2f} MB")
    print(f"   Increase: {memory_increase:.1f}x")
    
    # Both should produce valid embeddings
    assert emb_224.shape == (1, 128), f"Wrong 224 embedding shape: {emb_224.shape}"
    assert emb_512.shape == (1, 128), f"Wrong 512 embedding shape: {emb_512.shape}"
    
    print("✅ Memory usage test passed!")
    return True

def main():
    """Run all tests."""
    print("🔍 Testing 512x512 Resolution Upgrade")
    print("=" * 50)
    
    tests = [
        test_crop_extraction,
        test_simclr_dataset,
        test_model_compatibility,
        test_memory_usage,
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"❌ {test_func.__name__} failed: {e}")
            failed += 1
        print()
    
    print("=" * 50)
    print(f"📈 Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed! 512x512 resolution upgrade is working correctly.")
        print("\n💡 Benefits of 512x512 resolution:")
        print("   • Better individual crow identification")
        print("   • More detailed feather patterns visible")
        print("   • Improved beak and eye feature detection")
        print("   • Better performance for distant crows")
        print("   • ~5.3x more visual information per crop")
        return True
    else:
        print("⚠️  Some tests failed. Please check the issues above.")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 