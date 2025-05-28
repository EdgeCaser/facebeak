#!/usr/bin/env python3
"""
Simple test script to debug extraction issues.
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_basic_imports():
    """Test if all required modules can be imported."""
    print("🔧 Testing imports...")
    
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
        print(f"   CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   CUDA device: {torch.cuda.get_device_name(0)}")
    except ImportError as e:
        print(f"❌ PyTorch import failed: {e}")
        return False
    
    try:
        import torchvision
        print(f"✅ TorchVision: {torchvision.__version__}")
    except ImportError as e:
        print(f"❌ TorchVision import failed: {e}")
        return False
    
    try:
        from ultralytics import YOLO
        print(f"✅ Ultralytics YOLO imported")
    except ImportError as e:
        print(f"❌ Ultralytics import failed: {e}")
        return False
    
    try:
        from detection import detect_crows_parallel
        print(f"✅ Detection module imported")
    except ImportError as e:
        print(f"❌ Detection module import failed: {e}")
        return False
    
    try:
        from crow_tracking import CrowTracker
        print(f"✅ CrowTracker imported")
    except ImportError as e:
        print(f"❌ CrowTracker import failed: {e}")
        return False
    
    return True

def test_model_loading():
    """Test if detection models can be loaded."""
    print("\n🤖 Testing model loading...")
    
    try:
        from ultralytics import YOLO
        print("Loading YOLO model...")
        yolo_model = YOLO('yolov8s.pt')
        print("✅ YOLO model loaded successfully")
    except Exception as e:
        print(f"❌ YOLO model loading failed: {e}")
        return False
    
    try:
        import torchvision
        print("Loading Faster R-CNN model...")
        faster_rcnn_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        )
        faster_rcnn_model.eval()
        print("✅ Faster R-CNN model loaded successfully")
    except Exception as e:
        print(f"❌ Faster R-CNN model loading failed: {e}")
        return False
    
    return True

def test_detection_on_dummy_image():
    """Test detection on a dummy image."""
    print("\n🖼️ Testing detection on dummy image...")
    
    try:
        from detection import detect_crows_parallel
        
        # Create a dummy image (blue sky with a dark bird-like shape)
        dummy_image = np.ones((480, 640, 3), dtype=np.uint8) * 135  # Blue sky
        
        # Add a dark bird-like shape
        cv2.ellipse(dummy_image, (320, 240), (30, 15), 0, 0, 360, (50, 50, 50), -1)
        
        print("Running detection on dummy image...")
        detections = detect_crows_parallel([dummy_image], score_threshold=0.1)
        
        print(f"✅ Detection completed")
        print(f"   Detections found: {len(detections[0])}")
        
        for i, det in enumerate(detections[0]):
            print(f"   Detection {i+1}: score={det['score']:.3f}, class={det['class']}, model={det['model']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Detection test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_video_file(video_path):
    """Test a specific video file."""
    print(f"\n🎥 Testing video file: {video_path}")
    
    if not os.path.exists(video_path):
        print(f"❌ Video file not found: {video_path}")
        return False
    
    # Test video reading
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Cannot open video: {video_path}")
        return False
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"✅ Video info:")
    print(f"   Frames: {total_frames}")
    print(f"   FPS: {fps}")
    print(f"   Resolution: {width}x{height}")
    
    # Test reading a few frames
    frames_to_test = min(5, total_frames)
    frames = []
    
    for i in range(frames_to_test):
        frame_num = int(i * total_frames / frames_to_test)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    
    cap.release()
    
    if not frames:
        print("❌ Could not read any frames")
        return False
    
    print(f"✅ Read {len(frames)} frames successfully")
    
    # Test detection on these frames
    try:
        from detection import detect_crows_parallel
        print("Running detection on video frames...")
        
        detections = detect_crows_parallel(frames, score_threshold=0.1)
        total_detections = sum(len(d) for d in detections)
        
        print(f"✅ Detection completed")
        print(f"   Total detections: {total_detections}")
        
        for i, frame_dets in enumerate(detections):
            if frame_dets:
                print(f"   Frame {i}: {len(frame_dets)} detections")
                for j, det in enumerate(frame_dets):
                    print(f"     Detection {j+1}: score={det['score']:.3f}, class={det['class']}, model={det['model']}")
        
        return total_detections > 0
        
    except Exception as e:
        print(f"❌ Detection on video failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("🔍 EXTRACTION DEBUGGING TOOL")
    print("=" * 40)
    
    # Test 1: Imports
    imports_ok = test_basic_imports()
    if not imports_ok:
        print("\n❌ Import test failed. Please check your environment setup.")
        return
    
    # Test 2: Model loading
    models_ok = test_model_loading()
    if not models_ok:
        print("\n❌ Model loading failed. Please check your model files.")
        return
    
    # Test 3: Detection on dummy image
    dummy_ok = test_detection_on_dummy_image()
    if not dummy_ok:
        print("\n❌ Detection test failed. There may be an issue with the detection pipeline.")
        return
    
    # Test 4: Video file (if provided)
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
        video_ok = test_video_file(video_path)
        
        if not video_ok:
            print(f"\n❌ Video test failed for {video_path}")
            print("\n💡 Possible issues:")
            print("- Video format not supported")
            print("- Video is corrupted")
            print("- No birds/crows visible in the tested frames")
            print("- Detection confidence threshold too high")
        else:
            print(f"\n✅ Video test passed for {video_path}")
    else:
        print("\n💡 To test a specific video file, run:")
        print("python test_extraction.py path/to/your/video.mp4")
    
    print("\n📋 SUMMARY:")
    print(f"Imports: {'✅' if imports_ok else '❌'}")
    print(f"Models: {'✅' if models_ok else '❌'}")
    print(f"Detection: {'✅' if dummy_ok else '❌'}")
    
    if imports_ok and models_ok and dummy_ok:
        print("\n✅ Basic functionality works!")
        print("\n💡 If extraction still isn't working, try:")
        print("1. Lower the confidence threshold (--min-confidence 0.1)")
        print("2. Check that your videos actually contain birds")
        print("3. Run with verbose logging to see what's happening")
        print("4. Test with a known good video file")
    
if __name__ == "__main__":
    main() 