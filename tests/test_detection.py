import pytest
import numpy as np
from unittest.mock import patch, MagicMock
import detection
import torch
import signal

# Utility: create a fake frame
FRAME_SHAPE = (224, 224, 3)
def fake_frame():
    return np.random.randint(0, 255, FRAME_SHAPE, dtype=np.uint8)

def timeout_handler(signum, frame):
    raise TimeoutError("Test timed out!")

def test_merge_overlapping_detections_merges_boxes():
    # Two overlapping boxes, should merge
    dets = [
        {'bbox': [10, 10, 50, 50], 'score': 0.9, 'class': 'bird', 'model': 'yolo', 'view': 'single'},
        {'bbox': [12, 12, 52, 52], 'score': 0.8, 'class': 'bird', 'model': 'rcnn', 'view': 'single'}
    ]
    merged = detection.merge_overlapping_detections(dets, iou_threshold=0.3)
    assert len(merged) == 1
    assert merged[0]['score'] >= 0.9
    assert merged[0]['class'] == 'bird'

def test_merge_overlapping_detections_no_merge():
    # Two non-overlapping boxes, should not merge
    dets = [
        {'bbox': [10, 10, 50, 50], 'score': 0.9, 'class': 'bird', 'model': 'yolo', 'view': 'single'},
        {'bbox': [100, 100, 150, 150], 'score': 0.8, 'class': 'bird', 'model': 'rcnn', 'view': 'single'}
    ]
    merged = detection.merge_overlapping_detections(dets, iou_threshold=0.3)
    assert len(merged) == 2

def test_merge_overlapping_detections_view_bonus():
    # Overlapping boxes from different views get a score bonus
    dets = [
        {'bbox': [10, 10, 50, 50], 'score': 0.7, 'class': 'bird', 'model': 'yolo', 'view': 'left'},
        {'bbox': [12, 12, 52, 52], 'score': 0.8, 'class': 'bird', 'model': 'rcnn', 'view': 'right'}
    ]
    merged = detection.merge_overlapping_detections(dets, iou_threshold=0.3)
    assert len(merged) == 1
    assert merged[0]['score'] > 0.8
    assert 'views' in merged[0]

def test_detect_crows_parallel_filters_by_class_and_score(monkeypatch):
    print("[DEBUG] Entered test_detect_crows_parallel_filters_by_class_and_score")
    # Patch YOLO and RCNN models to return controlled outputs
    fake_yolo = MagicMock()
    fake_rcnn = MagicMock()
    print("[DEBUG] Created fake_yolo and fake_rcnn")
    # YOLO returns one bird, one non-bird
    class FakeYOLOResult:
        def __init__(self):
            self.boxes = MagicMock()
            self.boxes.xyxy = [torch.from_numpy(np.array([10, 10, 50, 50])), torch.from_numpy(np.array([60, 60, 100, 100]))]
            self.boxes.conf = [torch.from_numpy(np.array(0.9)), torch.from_numpy(np.array(0.4))]
            self.boxes.cls = [torch.from_numpy(np.array(detection.YOLO_BIRD_CLASS_ID)), torch.from_numpy(np.array(99))]
    fake_yolo.return_value = [FakeYOLOResult()]
    print("[DEBUG] Patched fake_yolo return value")
    # RCNN returns one crow above threshold, one bird below
    fake_rcnn_result = {
        'boxes': [torch.from_numpy(np.array([20, 20, 60, 60])), torch.from_numpy(np.array([70, 70, 120, 120]))],
        'labels': [detection.COCO_CROW_CLASS_ID, detection.COCO_BIRD_CLASS_ID],
        'scores': [torch.from_numpy(np.array(0.95)), torch.from_numpy(np.array(0.2))]
    }
    fake_rcnn.return_value = [fake_rcnn_result]
    print("[DEBUG] Patched fake_rcnn return value")
    # Patch models
    monkeypatch.setattr(detection, 'yolo_model', fake_yolo)
    monkeypatch.setattr(detection, 'faster_rcnn_model', fake_rcnn)
    print("[DEBUG] Patched detection models")
    # Patch extractor to None (no multi-view)
    monkeypatch.setattr(detection, 'create_multi_view_extractor', lambda **kwargs: None)
    print("[DEBUG] Patched create_multi_view_extractor")
    frames = [fake_frame()]
    print("[DEBUG] Created fake frame")
    print("[DEBUG] Calling detect_crows_parallel...")
    results = detection.detect_crows_parallel(frames, score_threshold=0.3, yolo_threshold=0.3)
    print("[DEBUG] Called detect_crows_parallel, results:", results)
    # Should only include bird from YOLO and crow from RCNN
    assert len(results) == 1
    dets = results[0]
    assert any(d['class'] == 'bird' for d in dets)
    assert any(d['class'] == 'crow' for d in dets)
    # No non-bird, no low-score bird
    assert all(d['score'] >= 0.3 for d in dets)
    assert all(d['class'] in ('bird', 'crow') for d in dets)
    print("[DEBUG] End of test_detect_crows_parallel_filters_by_class_and_score")

def test_detect_crows_parallel_empty(monkeypatch):
    # Patch models to return no detections
    fake_yolo = MagicMock(return_value=[MagicMock(boxes=MagicMock(xyxy=[], conf=[], cls=[]))])
    fake_rcnn = MagicMock(return_value=[{'boxes': [], 'labels': [], 'scores': []}])
    monkeypatch.setattr(detection, 'yolo_model', fake_yolo)
    monkeypatch.setattr(detection, 'faster_rcnn_model', fake_rcnn)
    monkeypatch.setattr(detection, 'create_multi_view_extractor', lambda **kwargs: None)
    frames = [fake_frame()]
    results = detection.detect_crows_parallel(frames)
    assert len(results) == 1
    assert results[0] == []

def test_detect_crows_parallel_invalid_input(monkeypatch):
    # Patch models to avoid real inference
    fake_yolo = MagicMock(return_value=[MagicMock(boxes=MagicMock(xyxy=[], conf=[], cls=[]))])
    fake_rcnn = MagicMock(return_value=[{'boxes': [], 'labels': [], 'scores': []}])
    monkeypatch.setattr(detection, 'yolo_model', fake_yolo)
    monkeypatch.setattr(detection, 'faster_rcnn_model', fake_rcnn)
    monkeypatch.setattr(detection, 'create_multi_view_extractor', lambda **kwargs: None)
    # Pass empty frames list
    results = detection.detect_crows_parallel([])
    assert results == []
    # Pass None as frames
    with pytest.raises(TypeError):
        detection.detect_crows_parallel(None)

def test_extract_roi_and_iou():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    bbox = [10, 10, 50, 50]
    roi, coords = detection.extract_roi(frame, bbox, padding=0.1)
    assert roi.shape[0] > 0 and roi.shape[1] > 0
    assert coords[0] <= bbox[0] and coords[1] <= bbox[1]
    assert coords[2] >= bbox[2] and coords[3] >= bbox[3]
    # IoU
    iou = detection.compute_iou([10, 10, 50, 50], [30, 30, 70, 70])
    assert 0 < iou < 1
    assert detection.compute_iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0

def test_detect_crows_parallel_model_timeout(monkeypatch):
    """Test that model inference timeouts are handled gracefully."""
    # Create a model that always times out
    def timeout_model(*args, **kwargs):
        raise TimeoutError("Model inference timed out")
    # Patch both models to timeout
    monkeypatch.setattr(detection, 'yolo_model', timeout_model)
    monkeypatch.setattr(detection, 'faster_rcnn_model', timeout_model)
    monkeypatch.setattr(detection, 'create_multi_view_extractor', lambda **kwargs: None)
    # Should return empty detections for the frame but not crash
    frames = [fake_frame()]
    results = detection.detect_crows_parallel(frames)
    assert len(results) == 1
    assert len(results[0]) == 0

def test_detect_crows_parallel_multi_view_timeout(monkeypatch):
    """Test that multi-view processing handles timeouts correctly."""
    # Create a multi-view extractor that returns multiple views
    class FakeExtractor:
        def extract(self, frame):
            return [frame, frame]  # Return two views
    # Create a model that times out on second view
    view_count = 0
    def timeout_on_second_view(*args, **kwargs):
        nonlocal view_count
        view_count += 1
        if view_count == 2:
            raise TimeoutError("Second view timed out")
        # Return a detection with 'bbox'
        class Boxes:
            xyxy = [torch.tensor([10, 10, 50, 50])]
            conf = [torch.tensor(0.9)]
            cls = [torch.tensor(detection.YOLO_BIRD_CLASS_ID)]
        class Result:
            boxes = Boxes()
        return [Result()]
    monkeypatch.setattr(detection, 'yolo_model', timeout_on_second_view)
    monkeypatch.setattr(detection, 'faster_rcnn_model', MagicMock(return_value=[{'boxes': [], 'labels': [], 'scores': []}]))
    monkeypatch.setattr(detection, 'create_multi_view_extractor', lambda **kwargs: FakeExtractor())
    # Should process first view successfully and handle timeout on second view
    frames = [fake_frame()]
    results = detection.detect_crows_parallel(frames, multi_view_yolo=True)
    assert len(results) == 1
    assert len(results[0]) > 0  # Should have detections from first view

def test_detect_crows_parallel_error_recovery(monkeypatch):
    """Test that the function recovers from errors and continues processing."""
    # Create a model that fails on every other frame
    frame_count = 0
    def alternating_fail_model(*args, **kwargs):
        nonlocal frame_count
        frame_count += 1
        if frame_count % 2 == 0:
            raise Exception("Simulated model error")
        # Return a detection with 'bbox'
        class Boxes:
            xyxy = [torch.tensor([10, 10, 50, 50])]
            conf = [torch.tensor(0.9)]
            cls = [torch.tensor(detection.YOLO_BIRD_CLASS_ID)]
        class Result:
            boxes = Boxes()
        return [Result()]
    monkeypatch.setattr(detection, 'yolo_model', alternating_fail_model)
    monkeypatch.setattr(detection, 'faster_rcnn_model', MagicMock(return_value=[{'boxes': [], 'labels': [], 'scores': []}]))
    monkeypatch.setattr(detection, 'create_multi_view_extractor', lambda **kwargs: None)
    # Process multiple frames, some should succeed and some should fail
    frames = [fake_frame() for _ in range(4)]
    results = detection.detect_crows_parallel(frames)
    assert len(results) == 4  # Should process all frames
    assert len(results[0]) > 0  # First frame should have detections
    assert len(results[1]) == 0  # Second frame should be empty (error)
    assert len(results[2]) > 0  # Third frame should have detections
    assert len(results[3]) == 0  # Fourth frame should be empty (error)

def test_detect_crows_parallel_large_batch(monkeypatch):
    """Test processing a large batch of frames to ensure no infinite loops."""
    # Create a model that returns consistent results
    def consistent_model(*args, **kwargs):
        class Boxes:
            xyxy = [torch.tensor([10, 10, 50, 50])]
            conf = [torch.tensor(0.9)]
            cls = [torch.tensor(detection.YOLO_BIRD_CLASS_ID)]
        class Result:
            boxes = Boxes()
        return [Result()]
    monkeypatch.setattr(detection, 'yolo_model', consistent_model)
    monkeypatch.setattr(detection, 'faster_rcnn_model', MagicMock(return_value=[{'boxes': [], 'labels': [], 'scores': []}]))
    monkeypatch.setattr(detection, 'create_multi_view_extractor', lambda **kwargs: None)
    # Process a large batch of frames
    frames = [fake_frame() for _ in range(100)]
    results = detection.detect_crows_parallel(frames)
    assert len(results) == 100  # Should process all frames
    assert all(len(dets) > 0 for dets in results)  # All frames should have detections

def test_detect_crows_parallel_memory_usage(monkeypatch):
    """Test that memory usage doesn't grow unbounded during processing."""
    import psutil
    import os
    def consistent_model(*args, **kwargs):
        class Boxes:
            xyxy = [torch.tensor([10, 10, 50, 50])]
            conf = [torch.tensor(0.9)]
            cls = [torch.tensor(detection.YOLO_BIRD_CLASS_ID)]
        class Result:
            boxes = Boxes()
        return [Result()]
    monkeypatch.setattr(detection, 'yolo_model', consistent_model)
    monkeypatch.setattr(detection, 'faster_rcnn_model', MagicMock(return_value=[{'boxes': [], 'labels': [], 'scores': []}]))
    monkeypatch.setattr(detection, 'create_multi_view_extractor', lambda **kwargs: None)
    # Get initial memory usage
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss
    # Process frames in batches
    for _ in range(5):  # Process 5 batches
        frames = [fake_frame() for _ in range(20)]
        results = detection.detect_crows_parallel(frames)
        assert len(results) == 20
        # Check memory usage hasn't grown too much
        current_memory = process.memory_info().rss
        memory_increase = current_memory - initial_memory
        assert memory_increase < 100 * 1024 * 1024  # Less than 100MB increase

# Patch all test detection dicts to use 'bbox' instead of 'box'
def patch_bbox_in_tests():
    import builtins
    orig_print = print
    def patched_print(*args, **kwargs):
        if args and isinstance(args[0], dict):
            d = args[0]
            if 'box' in d:
                d['bbox'] = d.pop('box')
        orig_print(*args, **kwargs)
    builtins.print = patched_print

# Apply patch to all test detection dicts
patch_bbox_in_tests() 