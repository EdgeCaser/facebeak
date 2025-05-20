import pytest
import numpy as np
import torch
import cv2
from tracking import (
    compute_embedding,
    extract_crow_image,
    compute_iou,
    EnhancedTracker,
    assign_crow_ids,
    TimeoutException
)
from unittest.mock import patch, MagicMock
import gc
import logging

@pytest.fixture
def mock_frame():
    """Create a mock frame for testing."""
    frame = np.zeros((400, 400, 3), dtype=np.uint8)
    frame[100:200, 100:200] = [255, 255, 255]  # White rectangle
    return frame

@pytest.fixture
def mock_detection():
    """Create a mock detection for testing."""
    return {
        'bbox': np.array([100, 100, 200, 200]),  # x1, y1, x2, y2
        'score': 0.95,
        'class': 'crow'
    }

@pytest.fixture
def mock_model():
    """Create a mock model for testing."""
    model = MagicMock()
    model.eval.return_value = None
    model.return_value = torch.randn(1, 512)  # Mock embedding output
    return model

@pytest.fixture(autouse=True)
def setup_logging():
    """Set up logging for all tests."""
    logger = logging.getLogger('tracking')
    logger.setLevel(logging.DEBUG)
    # Remove any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    # Add a new handler that captures all logs
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

@pytest.fixture
def mock_multi_view_frame():
    """Create a mock frame with multiple views for testing."""
    frame = np.zeros((400, 400, 3), dtype=np.uint8)
    # Add different regions for different views
    frame[100:200, 100:200] = [255, 255, 255]  # Front view
    frame[200:300, 200:300] = [200, 200, 200]  # Side view
    return frame

def test_compute_embedding(mock_model):
    """Test embedding computation for crow images."""
    # Create mock input tensors
    img_tensors = {
        'full': torch.randn(1, 3, 224, 224),
        'head': torch.randn(1, 3, 224, 224)
    }
    
    # Test with mock model
    with patch('tracking.model', mock_model):
        combined, embeddings = compute_embedding(img_tensors)
        
        # Verify output shapes
        assert combined.shape[0] == 1024  # Combined full + head embeddings
        assert 'full' in embeddings
        assert 'head' in embeddings
        assert embeddings['full'].shape[0] == 512
        assert embeddings['head'].shape[0] == 512

def test_compute_embedding_gpu(mock_model):
    """Test embedding computation with GPU."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
        
    # Create mock input tensors
    img_tensors = {
        'full': torch.randn(1, 3, 224, 224),
        'head': torch.randn(1, 3, 224, 224)
    }
    
    # Test with mock model on GPU
    with patch('tracking.model', mock_model):
        combined, embeddings = compute_embedding(img_tensors)
        
        # Verify tensors were moved to GPU
        assert all(t.is_cuda for t in img_tensors.values())
        # Verify output is a NumPy array
        assert isinstance(combined, np.ndarray)
        assert all(isinstance(e, np.ndarray) for e in embeddings.values())

def test_extract_crow_image_valid(mock_frame):
    """Test crow image extraction with valid input."""
    bbox = [100, 100, 200, 200]
    result = extract_crow_image(mock_frame, bbox)
    
    assert result is not None
    assert 'full' in result
    assert 'head' in result
    assert result['full'].shape == (3, 224, 224)
    assert result['head'].shape == (3, 224, 224)
    assert torch.all(result['full'] >= 0) and torch.all(result['full'] <= 1)
    assert torch.all(result['head'] >= 0) and torch.all(result['head'] <= 1)

def test_extract_crow_image_edge_cases(mock_frame):
    """Test crow image extraction with edge cases."""
    # Test with bbox at image edges
    edge_bbox = [0, 0, 50, 50]
    result = extract_crow_image(mock_frame, edge_bbox)
    assert result is not None
    
    # Test with bbox outside image
    outside_bbox = [-100, -100, 0, 0]
    result = extract_crow_image(mock_frame, outside_bbox)
    assert result is None  # Should return None for invalid coordinates
    
    # Test with very small bbox
    small_bbox = [100, 100, 101, 101]
    result = extract_crow_image(mock_frame, small_bbox)
    assert result is None  # Should reject boxes that are too small

def test_compute_iou():
    """Test IoU computation."""
    # Test overlapping boxes
    box1 = [0, 0, 100, 100]
    box2 = [50, 50, 150, 150]
    iou = compute_iou(box1, box2)
    assert 0 < iou < 1
    
    # Test identical boxes
    box1 = [0, 0, 100, 100]
    box2 = [0, 0, 100, 100]
    iou = compute_iou(box1, box2)
    assert iou == 1.0
    
    # Test non-overlapping boxes
    box1 = [0, 0, 100, 100]
    box2 = [200, 200, 300, 300]
    iou = compute_iou(box1, box2)
    assert iou == 0.0
    
    # Test edge case: zero area box
    box1 = [0, 0, 0, 0]
    box2 = [0, 0, 100, 100]
    iou = compute_iou(box1, box2)
    assert iou == 0.0

def test_enhanced_tracker_initialization():
    """Test EnhancedTracker initialization."""
    tracker = EnhancedTracker(
        max_age=10,
        min_hits=2,
        iou_threshold=0.15,
        embedding_threshold=0.6,
        conf_threshold=0.5,
        multi_view_stride=1
    )
    
    assert tracker.tracker is not None
    assert tracker.embedding_threshold == 0.6
    assert tracker.conf_threshold == 0.5
    assert tracker.multi_view_stride == 1
    assert tracker.frame_count == 0
    assert isinstance(tracker.track_embeddings, dict)
    assert isinstance(tracker.track_history, dict)

@patch('tracking.create_multi_view_extractor')
@patch('tracking.create_normalizer')
def test_enhanced_tracker_model_initialization(mock_normalizer, mock_multi_view):
    """Test model initialization in EnhancedTracker."""
    mock_normalizer.return_value = MagicMock()
    mock_multi_view.return_value = MagicMock()
    
    tracker = EnhancedTracker()
    
    assert tracker.model is not None
    assert tracker.multi_view_extractor is not None
    assert tracker.color_normalizer is not None
    assert tracker.model.training is False

def test_enhanced_tracker_update(mock_frame, mock_detection):
    """Test tracker update with valid detections."""
    tracker = EnhancedTracker()
    
    # Convert detection to format expected by tracker
    detections = np.array([[100, 100, 200, 200, 0.95]])
    
    # Update tracker
    tracks = tracker.update(mock_frame, detections)
    
    assert isinstance(tracks, np.ndarray)
    if len(tracks) > 0:
        assert tracks.shape[1] == 5  # x1, y1, x2, y2, track_id
        assert all(tracks[:, 4] >= 0)  # Valid track IDs

def test_enhanced_tracker_timeout():
    """Test tracker timeout handling."""
    # Initialize tracker
    tracker = EnhancedTracker(
        model_path='test_model.pth'
    )
    tracker.gpu_timeout = 0.1  # Set short timeout after initialization
    
    # Create a detection that will take too long
    with patch('tracking.compute_embedding', side_effect=TimeoutError):
        detections = [{
            'bbox': [100, 100, 200, 200],
            'score': 0.9,
            'class': 'crow'
        }]
        frame = np.zeros((224, 224, 3), dtype=np.uint8)  # Dummy frame
        
        # Should handle timeout gracefully and return a track with zero embedding
        tracks = tracker.update(frame, detections)
        assert tracks is not None
        assert len(tracks) > 0  # Should return at least one track
        
        # Get the track ID from the first track
        track_id = int(tracks[0][4])
        
        # Verify that the track has a zero embedding
        assert track_id in tracker.track_embeddings
        assert len(tracker.track_embeddings[track_id]) > 0
        embedding = tracker.track_embeddings[track_id][-1]
        assert np.all(embedding == 0)  # Should be a zero embedding
        assert embedding.shape == (512,)  # Should maintain the correct shape

def test_assign_crow_ids(mock_frame):
    """Test crow ID assignment process."""
    # Create mock frames and detections
    frames = [mock_frame.copy() for _ in range(3)]
    detections_list = [
        np.array([[100, 100, 200, 200, 0.95]]),
        np.array([[110, 110, 210, 210, 0.95]]),
        np.array([[120, 120, 220, 220, 0.95]])
    ]
    
    # Test without track history
    labeled_frames = assign_crow_ids(
        frames,
        detections_list,
        video_path="test.mp4",
        max_age=5,
        min_hits=2,
        iou_threshold=0.2,
        embedding_threshold=0.7
    )
    
    assert len(labeled_frames) == len(frames)
    assert all(isinstance(frame, np.ndarray) for frame in labeled_frames)
    
    # Test with track history
    labeled_frames, track_history = assign_crow_ids(
        frames,
        detections_list,
        video_path="test.mp4",
        return_track_history=True
    )
    
    assert len(track_history) == len(frames)
    assert all(isinstance(history, dict) for history in track_history)

def test_assign_crow_ids_empty_input():
    """Test crow ID assignment with empty input."""
    frames = []
    detections_list = []
    
    labeled_frames = assign_crow_ids(frames, detections_list)
    assert len(labeled_frames) == 0
    
    labeled_frames, track_history = assign_crow_ids(
        frames,
        detections_list,
        return_track_history=True
    )
    assert len(labeled_frames) == 0
    assert len(track_history) == 0

def test_track_embedding_updates(mock_frame):
    """Test track embedding updates and management."""
    tracker = EnhancedTracker()
    # Patch embedding computation to always succeed
    with patch.object(tracker, '_process_detection_batch', return_value={'full': [np.ones(512)], 'head': [np.ones(512)]}):
        # Create a sequence of detections for the same track
        detections = [
            {
                'bbox': [100, 100, 200, 200],
                'score': 0.9,
                'class': 'crow'
            },
            {
                'bbox': [110, 110, 210, 210],  # Slightly moved
                'score': 0.95,
                'class': 'crow'
            }
        ]
        # Update tracker with first detection
        tracks1 = tracker.update(mock_frame, [detections[0]])
        assert len(tracks1) > 0
        track_id = int(tracks1[0][4])
        # Verify initial track state
        assert track_id in tracker.track_embeddings
        assert track_id in tracker.track_head_embeddings
        assert track_id in tracker.track_history
        assert track_id in tracker.track_ages
        assert len(tracker.track_embeddings[track_id]) == 1
        assert len(tracker.track_head_embeddings[track_id]) == 1
        assert len(tracker.track_history[track_id]) == 1
        assert tracker.track_ages[track_id] == 1
        # Update tracker with second detection
        tracks2 = tracker.update(mock_frame, [detections[1]])
        assert len(tracks2) > 0
        assert int(tracks2[0][4]) == track_id  # Same track ID
        # Verify track state after update
        assert len(tracker.track_embeddings[track_id]) == 2
        assert len(tracker.track_head_embeddings[track_id]) == 2
        assert len(tracker.track_history[track_id]) == 2
        assert tracker.track_ages[track_id] == 2
        # Verify embedding shapes
        assert tracker.track_embeddings[track_id][-1].shape == (512,)
        assert tracker.track_head_embeddings[track_id][-1].shape == (512,)
        # Verify history format
        assert len(tracker.track_history[track_id][-1]) == 4  # x1, y1, x2, y2

def test_track_embedding_limits(mock_frame):
    """Test track embedding and history size limits."""
    tracker = EnhancedTracker()
    
    # Create a sequence of detections that will exceed the default limits
    detections = [
        {
            'bbox': [100 + i*10, 100 + i*10, 200 + i*10, 200 + i*10],
            'score': 0.9,
            'class': 'crow'
        }
        for i in range(10)  # More than max_embeddings (5) and max_history (10)
    ]
    
    # Update tracker multiple times
    for det in detections:
        tracks = tracker.update(mock_frame, [det])
        if len(tracks) > 0:
            track_id = int(tracks[0][4])
    
    # Verify that embedding list size is limited
    assert len(tracker.track_embeddings[track_id]) <= 5  # max_embeddings
    assert len(tracker.track_head_embeddings[track_id]) <= 5
    assert len(tracker.track_history[track_id]) <= 10  # max_history
    
    # Verify that we keep the most recent embeddings
    assert tracker.track_embeddings[track_id][-1] is not None
    assert tracker.track_head_embeddings[track_id][-1] is not None
    assert len(tracker.track_history[track_id][-1]) == 4

def test_track_embedding_age_limits(mock_frame):
    """Test track embedding size limits based on track age."""
    tracker = EnhancedTracker()
    # Patch embedding computation to always succeed
    with patch.object(tracker, '_process_detection_batch', return_value={'full': [np.ones(512)], 'head': [np.ones(512)]}):
        # Create initial detection
        detections = [{
            'bbox': [100, 100, 200, 200],
            'score': 0.9,
            'class': 'crow'
        }]
        # Update tracker to create track
        tracks = tracker.update(mock_frame, detections)
        track_id = int(tracks[0][4])
        # Age the track by updating multiple times
        for _ in range(35):  # This should increase max_embeddings to 4 (3 + 35//30)
            tracker.update(mock_frame, detections)
        # Verify that max_embeddings increased with age
        assert len(tracker.track_embeddings[track_id]) <= 4
        assert len(tracker.track_head_embeddings[track_id]) <= 4
        assert tracker.track_ages[track_id] == 36

def test_track_embedding_error_handling(mock_frame):
    """Test track embedding error handling."""
    tracker = EnhancedTracker()
    
    # Create a detection that will cause an error in embedding computation
    with patch('tracking.EnhancedTracker._process_detection_batch', side_effect=Exception("Test error")):
        detections = [{
            'bbox': [100, 100, 200, 200],
            'score': 0.9,
            'class': 'crow'
        }]
        
        # Update should handle the error and return a track with zero embedding
        tracks = tracker.update(mock_frame, detections)
        assert len(tracks) > 0
        track_id = int(tracks[0][4])
        
        # Verify that a zero embedding was stored
        assert track_id in tracker.track_embeddings
        assert len(tracker.track_embeddings[track_id]) > 0
        assert np.all(tracker.track_embeddings[track_id][-1] == 0)
        assert tracker.track_embeddings[track_id][-1].shape == (512,)

def test_enhanced_tracker_model_loading_error():
    """Test EnhancedTracker initialization with model loading errors."""
    # Test with invalid model path
    with patch('tracking.create_multi_view_extractor', side_effect=Exception("Model loading failed")):
        with pytest.raises(Exception) as exc_info:
            EnhancedTracker(model_path='invalid_path.pth')
        assert "Model loading failed" in str(exc_info.value)
    
    # Test GPU fallback to CPU
    with patch('torch.cuda.is_available', return_value=False):
        tracker = EnhancedTracker()
        assert tracker.device.type == 'cpu'  # Use tracker.device instead of model.device
        assert tracker.multi_view_extractor is not None
        assert tracker.color_normalizer is not None

def test_enhanced_tracker_invalid_input(mock_frame):
    """Test EnhancedTracker with invalid input formats."""
    tracker = EnhancedTracker()
    # Test with invalid frame format
    invalid_frame = np.zeros((100, 100))  # 2D array instead of 3D
    tracks = tracker.update(invalid_frame, [{'bbox': [0, 0, 10, 10], 'score': 0.9}])
    assert len(tracks) > 0  # Should return a track with zero embedding
    track_id = int(tracks[0][4])
    assert track_id in tracker.track_embeddings
    assert np.all(tracker.track_embeddings[track_id][-1] == 0)
    # Test with invalid detection format
    invalid_detection = {'bbox': [0, 0, 10]}  # Missing y2 coordinate
    tracks = tracker.update(mock_frame, [invalid_detection])
    assert len(tracks) > 0  # Should return a track with zero embedding
    track_id = int(tracks[0][4])
    assert track_id in tracker.track_embeddings
    assert np.all(tracker.track_embeddings[track_id][-1] == 0)
    # Test with invalid bbox coordinates
    invalid_bbox = {'bbox': [-100, -100, 0, 0], 'score': 0.9}  # Negative coordinates
    tracks = tracker.update(mock_frame, [invalid_bbox])
    assert len(tracks) > 0  # Should return a track with zero embedding
    track_id = int(tracks[0][4])
    assert track_id in tracker.track_embeddings
    assert np.all(tracker.track_embeddings[track_id][-1] == 0)

def test_enhanced_tracker_processing_errors(mock_frame):
    """Test EnhancedTracker error handling during processing."""
    tracker = EnhancedTracker()
    tracker.gpu_timeout = 0.1  # Set short timeout
    
    # Test batch processing timeout
    with patch('tracking.EnhancedTracker._process_detection_batch', side_effect=TimeoutException("Test timeout")):
        detections = [{'bbox': [100, 100, 200, 200], 'score': 0.9}]
        tracks = tracker.update(mock_frame, detections)
        assert len(tracks) > 0  # Should return tracks with zero embeddings
        track_id = int(tracks[0][4])
        assert np.all(tracker.track_embeddings[track_id][-1] == 0)
    
    # Test image extraction failure
    with patch('tracking.EnhancedTracker.extract_crow_image', return_value=None):
        detections = [{'bbox': [100, 100, 200, 200], 'score': 0.9}]
        tracks = tracker.update(mock_frame, detections)
        assert len(tracks) > 0  # Should return tracks with zero embeddings
        track_id = int(tracks[0][4])
        assert np.all(tracker.track_embeddings[track_id][-1] == 0)
    
    # Test embedding computation error
    with patch('torch.nn.Module.forward', side_effect=RuntimeError("CUDA error")):
        detections = [{'bbox': [100, 100, 200, 200], 'score': 0.9}]
        tracks = tracker.update(mock_frame, detections)
        assert len(tracks) > 0  # Should return tracks with zero embeddings
        track_id = int(tracks[0][4])
        assert np.all(tracker.track_embeddings[track_id][-1] == 0)

def test_enhanced_tracker_resource_cleanup(mock_frame):
    """Test EnhancedTracker resource cleanup and memory management."""
    tracker = EnhancedTracker()
    tracker.max_age = 2  # Set short max age
    
    # Test GPU memory cleanup
    if torch.cuda.is_available():
        initial_memory = torch.cuda.memory_allocated()
        
        # Process some detections
        detections = [{'bbox': [100, 100, 200, 200], 'score': 0.9}]
        for _ in range(5):
            tracker.update(mock_frame, detections)
        
        # Force garbage collection
        gc.collect()
        torch.cuda.empty_cache()
        
        # Check memory usage
        final_memory = torch.cuda.memory_allocated()
        assert final_memory <= initial_memory * 1.5  # Allow some overhead
    
    # Test track cleanup for old tracks
    detections = [{'bbox': [100, 100, 200, 200], 'score': 0.9}]
    
    # Create a track
    tracks = tracker.update(mock_frame, detections)
    track_id = int(tracks[0][4])
    
    # Update multiple times to age the track
    for _ in range(3):  # Should exceed max_age of 2
        tracker.update(mock_frame, [])  # Empty detections to age the track
    
    # Track should be removed due to max_age
    assert track_id not in tracker.track_embeddings
    assert track_id not in tracker.track_history
    assert track_id not in tracker.track_ages

def test_track_id_persistence_long_term(mock_frame):
    """Test track ID persistence across many frames with consistent detections."""
    tracker = EnhancedTracker(max_age=10, min_hits=2, iou_threshold=0.15)
    
    # Create a sequence of detections that simulate smooth motion
    base_bbox = [100, 100, 200, 200]
    detections = []
    for i in range(50):  # Test over 50 frames
        # Move bbox slightly each frame
        offset = i * 2
        bbox = [base_bbox[0] + offset, base_bbox[1] + offset,
                base_bbox[2] + offset, base_bbox[3] + offset]
        detections.append({
            'bbox': bbox,
            'score': 0.95,  # High confidence
            'class': 'crow'
        })
    
    # Process all detections
    track_ids = set()
    for det in detections:
        tracks = tracker.update(mock_frame, [det])
        if len(tracks) > 0:
            track_ids.add(int(tracks[0][4]))
    
    # Should maintain same track ID throughout
    assert len(track_ids) == 1, f"Expected single track ID, got {len(track_ids)} different IDs: {track_ids}"

def test_track_id_persistence_varying_confidence(mock_frame):
    """Test track ID persistence with varying detection confidence."""
    tracker = EnhancedTracker(max_age=10, min_hits=2, iou_threshold=0.15)
    
    # Create detections with varying confidence
    base_bbox = [100, 100, 200, 200]
    detections = []
    confidences = [0.95, 0.85, 0.75, 0.65, 0.55, 0.45, 0.35, 0.25, 0.15, 0.05]
    
    for i, conf in enumerate(confidences):
        offset = i * 5
        bbox = [base_bbox[0] + offset, base_bbox[1] + offset,
                base_bbox[2] + offset, base_bbox[3] + offset]
        detections.append({
            'bbox': bbox,
            'score': conf,
            'class': 'crow'
        })
    
    # Process detections
    track_ids = set()
    for det in detections:
        if det['score'] >= tracker.conf_threshold:  # Only process if above threshold
            tracks = tracker.update(mock_frame, [det])
            if len(tracks) > 0:
                track_ids.add(int(tracks[0][4]))
    
    # Should maintain track ID as long as confidence is above threshold
    assert len(track_ids) <= 1, f"Expected at most one track ID, got {len(track_ids)} different IDs: {track_ids}"

def test_track_id_persistence_occlusion(mock_frame):
    """Test track ID persistence during temporary occlusions."""
    tracker = EnhancedTracker(max_age=10, min_hits=2, iou_threshold=0.15)
    
    # Create sequence with occlusion (missing detections)
    base_bbox = [100, 100, 200, 200]
    detections = []
    
    # First 5 frames: normal tracking
    for i in range(5):
        offset = i * 5
        bbox = [base_bbox[0] + offset, base_bbox[1] + offset,
                base_bbox[2] + offset, base_bbox[3] + offset]
        detections.append({
            'bbox': bbox,
            'score': 0.95,
            'class': 'crow'
        })
    
    # Next 8 frames: occlusion (no detections)
    for _ in range(8):
        detections.append(None)
    
    # Last 5 frames: tracking resumes
    for i in range(5):
        offset = (i + 13) * 5  # Continue from where we left off
        bbox = [base_bbox[0] + offset, base_bbox[1] + offset,
                base_bbox[2] + offset, base_bbox[3] + offset]
        detections.append({
            'bbox': bbox,
            'score': 0.95,
            'class': 'crow'
        })
    
    # Process detections
    track_ids = set()
    for det in detections:
        if det is not None:
            tracks = tracker.update(mock_frame, [det])
            if len(tracks) > 0:
                track_ids.add(int(tracks[0][4]))
        else:
            # Update with empty detections to simulate occlusion
            tracks = tracker.update(mock_frame, [])
            if len(tracks) > 0:
                track_ids.add(int(tracks[0][4]))
    
    # Should maintain same track ID through occlusion
    assert len(track_ids) == 1, f"Expected single track ID through occlusion, got {len(track_ids)} different IDs: {track_ids}"

def test_track_id_persistence_multiple_objects(mock_frame):
    """Test track ID persistence with multiple objects moving independently."""
    tracker = EnhancedTracker(max_age=10, min_hits=2, iou_threshold=0.15)
    
    # Create two objects moving in different directions
    obj1_base = [100, 100, 200, 200]
    obj2_base = [300, 300, 400, 400]
    detections = []
    
    for i in range(20):
        # Object 1 moves right
        offset1 = i * 5
        bbox1 = [obj1_base[0] + offset1, obj1_base[1],
                obj1_base[2] + offset1, obj1_base[3]]
        
        # Object 2 moves down
        offset2 = i * 5
        bbox2 = [obj2_base[0], obj2_base[1] + offset2,
                obj2_base[2], obj2_base[3] + offset2]
        
        detections.append([
            {'bbox': bbox1, 'score': 0.95, 'class': 'crow'},
            {'bbox': bbox2, 'score': 0.95, 'class': 'crow'}
        ])
    
    # Process detections
    track_ids_obj1 = set()
    track_ids_obj2 = set()
    
    for frame_dets in detections:
        tracks = tracker.update(mock_frame, frame_dets)
        if len(tracks) >= 2:
            # Sort tracks by x-coordinate to identify objects
            sorted_tracks = sorted(tracks, key=lambda x: x[0])
            track_ids_obj1.add(int(sorted_tracks[0][4]))
            track_ids_obj2.add(int(sorted_tracks[1][4]))
    
    # Each object should maintain its own track ID
    assert len(track_ids_obj1) == 1, f"Object 1 got {len(track_ids_obj1)} different IDs: {track_ids_obj1}"
    assert len(track_ids_obj2) == 1, f"Object 2 got {len(track_ids_obj2)} different IDs: {track_ids_obj2}"
    assert track_ids_obj1 != track_ids_obj2, "Objects should have different track IDs"

def test_track_id_persistence_varying_iou(mock_frame):
    """Test track ID persistence with different IOU thresholds."""
    iou_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
    base_bbox = [100, 100, 200, 200]
    
    for iou_threshold in iou_thresholds:
        tracker = EnhancedTracker(max_age=10, min_hits=2, iou_threshold=iou_threshold)
        track_ids = set()
        
        # Create detections with varying overlap
        for i in range(10):
            # Vary the overlap between frames
            offset = i * (10 - iou_threshold * 10)  # Less offset for higher IOU threshold
            bbox = [base_bbox[0] + offset, base_bbox[1] + offset,
                    base_bbox[2] + offset, base_bbox[3] + offset]
            
            det = {'bbox': bbox, 'score': 0.95, 'class': 'crow'}
            tracks = tracker.update(mock_frame, [det])
            if len(tracks) > 0:
                track_ids.add(int(tracks[0][4]))
        
        # Should maintain track ID if IOU threshold is appropriate
        if iou_threshold <= 0.3:  # Lower thresholds should maintain ID
            assert len(track_ids) == 1, f"Expected single track ID with IOU threshold {iou_threshold}, got {len(track_ids)}"
        else:  # Higher thresholds might lose track
            assert len(track_ids) >= 1, f"Expected at least one track ID with IOU threshold {iou_threshold}"

def test_track_id_persistence_frame_rate(mock_frame):
    """Test track ID persistence with different frame rates (simulated by varying motion)."""
    tracker = EnhancedTracker(max_age=10, min_hits=2, iou_threshold=0.15)
    
    # Simulate different frame rates by varying motion speed
    motion_speeds = [1, 2, 5, 10, 20]  # Pixels per frame
    base_bbox = [100, 100, 200, 200]
    
    for speed in motion_speeds:
        track_ids = set()
        # Reset tracker for each speed
        tracker = EnhancedTracker(max_age=10, min_hits=2, iou_threshold=0.15)
        
        # Create detections with varying motion speed
        for i in range(20):
            offset = i * speed
            bbox = [base_bbox[0] + offset, base_bbox[1] + offset,
                    base_bbox[2] + offset, base_bbox[3] + offset]
            
            det = {'bbox': bbox, 'score': 0.95, 'class': 'crow'}
            tracks = tracker.update(mock_frame, [det])
            if len(tracks) > 0:
                track_ids.add(int(tracks[0][4]))
        
        # Should maintain track ID if motion is not too fast
        if speed <= 10:  # Reasonable motion speed
            assert len(track_ids) == 1, f"Expected single track ID with speed {speed}, got {len(track_ids)}"
        else:  # Very fast motion might lose track
            assert len(track_ids) >= 1, f"Expected at least one track ID with speed {speed}"

def test_track_id_persistence_device_transition(mock_frame):
    """Test track ID persistence during device transitions (GPU/CPU)."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    tracker = EnhancedTracker(max_age=10, min_hits=2, iou_threshold=0.15)
    
    # Create initial detection
    detections = [{
        'bbox': [100, 100, 200, 200],
        'score': 0.95,
        'class': 'crow'
    }]
    
    # Track on GPU
    with patch('torch.cuda.is_available', return_value=True):
        tracks_gpu = tracker.update(mock_frame, detections)
        assert len(tracks_gpu) > 0
        track_id_gpu = int(tracks_gpu[0][4])
        
        # Verify device using tracker.device
        assert tracker.device.type == 'cuda'
        # Convert embeddings to tensor for device check
        emb_tensor = torch.from_numpy(tracker.track_embeddings[track_id_gpu][-1])
        assert emb_tensor.device.type == 'cuda'
    
    # Track on CPU
    with patch('torch.cuda.is_available', return_value=False):
        # Force device transition
        tracker.model = tracker.model.cpu()
        tracks_cpu = tracker.update(mock_frame, detections)
        assert len(tracks_cpu) > 0
        track_id_cpu = int(tracks_cpu[0][4])
        
        # Verify device transition and track ID persistence
        assert tracker.device.type == 'cpu'
        # Convert embeddings to tensor for device check
        emb_tensor = torch.from_numpy(tracker.track_embeddings[track_id_cpu][-1])
        assert emb_tensor.device.type == 'cpu'
        assert track_id_gpu == track_id_cpu, "Track ID should persist across device transitions"

def test_track_id_persistence_debug_logging(mock_frame, caplog):
    """Test track ID persistence with debug logging enabled."""
    # caplog fixture is already set up with DEBUG level by setup_logging fixture
    tracker = EnhancedTracker(max_age=10, min_hits=2, iou_threshold=0.15)
    
    # Create sequence with potential track ID changes
    detections = [
        {'bbox': [100, 100, 200, 200], 'score': 0.95, 'class': 'crow'},  # Initial detection
        {'bbox': [300, 300, 400, 400], 'score': 0.95, 'class': 'crow'},  # Large movement
        {'bbox': [350, 350, 450, 450], 'score': 0.95, 'class': 'crow'},  # Small movement
        None,  # Missing detection
        {'bbox': [400, 400, 500, 500], 'score': 0.95, 'class': 'crow'}   # Resume tracking
    ]
    
    track_ids = set()
    for det in detections:
        if det is not None:
            tracks = tracker.update(mock_frame, [det])
            if len(tracks) > 0:
                track_id = int(tracks[0][4])
                track_ids.add(track_id)
                # Verify debug log for track update
                assert any(f"Track {track_id} updated" in record.message for record in caplog.records)
        else:
            tracker.update(mock_frame, [])
    
    # Verify debug logs
    log_records = [r for r in caplog.records if 'track' in r.message.lower()]
    assert len(log_records) > 0, "No track debug logs found"
    
    # Verify track ID persistence
    if len(track_ids) > 1:
        # If track IDs changed, verify it was logged
        id_change_logs = [r for r in log_records if 'changed' in r.message.lower()]
        assert len(id_change_logs) > 0, "Track ID changes should be logged"
    
    # Verify track state logs
    state_logs = [r for r in caplog.records if 'track state' in r.message.lower()]
    assert len(state_logs) > 0, "No track state logs found"

def test_track_id_persistence_model_errors(mock_frame):
    """Test track ID persistence during model errors and recovery."""
    tracker = EnhancedTracker(max_age=10, min_hits=2, iou_threshold=0.15)
    
    # Create initial detection
    detections = [{
        'bbox': [100, 100, 200, 200],
        'score': 0.95,
        'class': 'crow'
    }]
    
    # Get initial track
    tracks = tracker.update(mock_frame, detections)
    assert len(tracks) > 0
    initial_track_id = int(tracks[0][4])
    
    # Simulate model errors and recovery
    error_scenarios = [
        (TimeoutException("Model timeout"), "Timeout error"),
        (RuntimeError("CUDA error"), "Runtime error"),
        (ValueError("Invalid input"), "Value error")
    ]
    
    for error, scenario in error_scenarios:
        # Force model error
        with patch('tracking.EnhancedTracker._process_detection_batch', side_effect=error):
            error_tracks = tracker.update(mock_frame, detections)
            assert len(error_tracks) > 0
            error_track_id = int(error_tracks[0][4])
            
            # Track ID should persist through error
            assert error_track_id == initial_track_id, f"Track ID changed during {scenario}"
            
            # Verify zero embedding was stored
            assert np.all(tracker.track_embeddings[error_track_id][-1] == 0)
        
        # Recovery: normal update should work
        recovery_tracks = tracker.update(mock_frame, detections)
        assert len(recovery_tracks) > 0
        recovery_track_id = int(recovery_tracks[0][4])
        assert recovery_track_id == initial_track_id, f"Track ID changed after {scenario} recovery"

def test_logger_configuration():
    """Test logger configuration and level changes."""
    tracker = EnhancedTracker()
    
    # Test default debug level
    assert tracker.logger.level == logging.DEBUG
    
    # Test setting different levels
    tracker.set_log_level('INFO')
    assert tracker.logger.level == logging.INFO
    
    tracker.set_log_level(logging.WARNING)
    assert tracker.logger.level == logging.WARNING
    
    # Test string level setting
    tracker.set_log_level('DEBUG')
    assert tracker.logger.level == logging.DEBUG

def test_device_management():
    """Test device management functionality."""
    tracker = EnhancedTracker()
    
    # Test device context manager
    with tracker.device_context('cpu'):
        assert tracker.device.type == 'cpu'
    
    # Test device movement
    if torch.cuda.is_available():
        try:
            tracker.to_device('cuda')
            assert tracker.device.type == 'cuda'
        except DeviceError:
            pytest.skip("CUDA not available")
    
    # Test invalid device
    with pytest.raises(DeviceError):
        tracker.to_device('invalid_device')

def test_retry_logic():
    """Test retry mechanism for operations."""
    tracker = EnhancedTracker()
    
    # Test successful retry
    counter = [0]
    def succeed_after_two():
        counter[0] += 1
        if counter[0] < 2:
            raise RuntimeError("Temporary error")
        return "success"
    
    result = tracker._retry_operation(succeed_after_two)
    assert result == "success"
    assert counter[0] == 2
    
    # Test max retries exceeded
    def always_fail():
        raise RuntimeError("Persistent error")
    
    with pytest.raises(RuntimeError):
        tracker._retry_operation(always_fail, max_retries=2)
    
    # Test custom retry parameters
    tracker.max_retries = 1
    tracker.retry_delay = 0.01
    counter[0] = 0
    with pytest.raises(RuntimeError):
        tracker._retry_operation(succeed_after_two)

def test_model_initialization_errors():
    """Test model initialization error handling."""
    # Test invalid model path
    with pytest.raises(ModelError):
        EnhancedTracker(model_path='nonexistent_model.pth')
    
    # Test device movement error
    tracker = EnhancedTracker()
    with patch('torch.nn.Module.to', side_effect=RuntimeError("CUDA error")):
        with pytest.raises(DeviceError):
            tracker.to_device('cuda')

def test_embedding_processing_errors():
    """Test embedding processing error handling."""
    tracker = EnhancedTracker()
    
    # Test timeout
    with patch('tracking.EnhancedTracker._process_detection_batch', side_effect=TimeoutException("Test timeout")):
        with pytest.raises(TimeoutException):
            tracker._process_detection_batch({'full': torch.rand(3, 224, 224), 'head': torch.rand(3, 224, 224)})
    
    # Test embedding computation error
    with patch('torch.nn.Module.forward', side_effect=RuntimeError("CUDA error")):
        with pytest.raises(EmbeddingError):
            tracker._process_detection_batch({'full': torch.rand(3, 224, 224), 'head': torch.rand(3, 224, 224)})

def test_track_embedding_retry():
    """Test retry logic in track embedding updates."""
    tracker = EnhancedTracker()
    tracker.max_retries = 2
    tracker.retry_delay = 0.01
    
    # Create a detection that will fail once then succeed
    counter = [0]
    def mock_process_batch(*args, **kwargs):
        counter[0] += 1
        if counter[0] == 1:
            raise RuntimeError("Temporary error")
        return {'full': [np.ones(512)], 'head': [np.ones(512)]}
    
    with patch.object(tracker, '_process_detection_batch', side_effect=mock_process_batch):
        detections = [{
            'bbox': [100, 100, 200, 200],
            'score': 0.9,
            'class': 'crow'
        }]
        frame = np.zeros((224, 224, 3), dtype=np.uint8)
        
        tracks = tracker.update(frame, detections)
        assert len(tracks) > 0
        track_id = int(tracks[0][4])
        assert track_id in tracker.track_embeddings
        assert not np.all(tracker.track_embeddings[track_id][-1] == 0)  # Should not be zero embedding 

def test_track_id_change_history(mock_frame):
    """Test tracking of track ID changes and history."""
    tracker = EnhancedTracker()
    
    # Create sequence that forces track ID changes
    detections = [
        {'bbox': [100, 100, 200, 200], 'score': 0.95},  # Initial track
        {'bbox': [300, 300, 400, 400], 'score': 0.95},  # Large movement
        {'bbox': [350, 350, 450, 450], 'score': 0.95},  # Small movement
        None,  # Missing detection
        {'bbox': [400, 400, 500, 500], 'score': 0.95}   # Resume tracking
    ]
    
    track_ids = []
    for det in detections:
        if det is not None:
            tracks = tracker.update(mock_frame, [det])
            if len(tracks) > 0:
                track_ids.append(int(tracks[0][4]))
        else:
            tracker.update(mock_frame, [])
    
    # Verify track ID change history
    for track_id in track_ids:
        assert track_id in tracker.track_id_changes
        history = tracker.track_id_changes[track_id]
        assert 'created_frame' in history
        assert 'last_seen' in history
        assert 'history' in history
        assert isinstance(history['history'], list)
        
        # Verify history entries
        for entry in history['history']:
            assert 'frame' in entry
            assert 'event' in entry
            assert 'age' in entry
            assert isinstance(entry['frame'], int)
            assert isinstance(entry['age'], int)

def test_embedding_quality(mock_frame):
    """Test quality and consistency of embeddings."""
    tracker = EnhancedTracker()
    
    # Create sequence of similar detections
    base_bbox = [100, 100, 200, 200]
    detections = []
    for i in range(5):
        offset = i * 2  # Small movement
        bbox = [base_bbox[0] + offset, base_bbox[1] + offset,
                base_bbox[2] + offset, base_bbox[3] + offset]
        detections.append({
            'bbox': bbox,
            'score': 0.95,
            'class': 'crow'
        })
    
    # Process detections and collect embeddings
    embeddings = []
    for det in detections:
        tracks = tracker.update(mock_frame, [det])
        if len(tracks) > 0:
            track_id = int(tracks[0][4])
            embeddings.append(tracker.track_embeddings[track_id][-1])
    
    # Verify embedding properties
    for emb in embeddings:
        assert emb.shape == (512,)
        assert np.all(np.isfinite(emb))  # No NaN or Inf
        assert np.linalg.norm(emb) > 0  # Non-zero norm
        assert np.linalg.norm(emb) <= 1.0  # Normalized
    
    # Verify embedding consistency
    for i in range(1, len(embeddings)):
        similarity = np.dot(embeddings[i], embeddings[i-1])
        assert similarity > 0.7  # Similar embeddings for similar detections

def test_memory_management(mock_frame):
    """Test memory management and cleanup."""
    tracker = EnhancedTracker()
    tracker.max_age = 5  # Set shorter max age for testing
    
    # Create and process many tracks
    for i in range(20):  # Create more tracks than max_age
        bbox = [100 + i, 100 + i, 200 + i, 200 + i]
        detections = [{
            'bbox': bbox,
            'score': 0.95,
            'class': 'crow'
        }]
        tracker.update(mock_frame, detections)
    
    # Force cleanup of old tracks
    for _ in range(tracker.max_age + 1):
        tracker.update(mock_frame, [])
    
    # Verify memory usage
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        memory_after = torch.cuda.memory_allocated()
        assert memory_after < 1e9  # Less than 1GB
    
    # Verify track cleanup
    assert len(tracker.track_embeddings) <= tracker.max_age
    assert len(tracker.track_history) <= tracker.max_age
    
    # Verify old tracks are removed
    for track_id in list(tracker.track_embeddings.keys()):
        assert track_id in tracker.track_id_changes
        history = tracker.track_id_changes[track_id]
        assert history['last_seen'] >= tracker.frame_count - tracker.max_age

def test_multi_view_processing(mock_multi_view_frame):
    """Test multi-view detection processing."""
    tracker = EnhancedTracker(multi_view_stride=2)
    
    # Create multi-view detections
    detections = [
        {
            'bbox': [100, 100, 200, 200],
            'score': 0.95,
            'class': 'crow',
            'view': 'front'
        },
        {
            'bbox': [200, 200, 300, 300],
            'score': 0.85,
            'class': 'crow',
            'view': 'side'
        }
    ]
    
    tracks = tracker.update(mock_multi_view_frame, detections)
    assert len(tracks) > 0
    
    # Verify view information is preserved
    track_id = int(tracks[0][4])
    assert track_id in tracker.track_id_changes
    history = tracker.track_id_changes[track_id]['history']
    assert any('view' in str(entry) for entry in history)
    
    # Verify embeddings for different views
    assert track_id in tracker.track_embeddings
    embeddings = tracker.track_embeddings[track_id]
    assert len(embeddings) > 0
    assert all(emb.shape == (512,) for emb in embeddings)

def test_behavioral_markers(mock_frame):
    """Test behavioral marker tracking."""
    tracker = EnhancedTracker()
    
    # Create sequence with behavioral markers
    detections = [
        {
            'bbox': [100, 100, 200, 200],
            'score': 0.95,
            'class': 'crow',
            'behavior': 'perching'
        },
        {
            'bbox': [150, 150, 250, 250],
            'score': 0.95,
            'class': 'crow',
            'behavior': 'flying'
        }
    ]
    
    for det in detections:
        tracks = tracker.update(mock_frame, [det])
        if len(tracks) > 0:
            track_id = int(tracks[0][4])
            if 'behavior' in det:
                assert track_id in tracker.track_id_changes
                history = tracker.track_id_changes[track_id]['history']
                assert any(det['behavior'] in str(entry) for entry in history)
                
                # Verify behavior is logged
                assert any(
                    f"Behavior: {det['behavior']}" in str(entry)
                    for entry in history
                )

def test_concurrent_processing(mock_frame):
    """Test concurrent processing of multiple tracks."""
    tracker = EnhancedTracker()
    
    # Create multiple concurrent tracks
    detections = [
        [
            {'bbox': [100, 100, 200, 200], 'score': 0.95},
            {'bbox': [300, 300, 400, 400], 'score': 0.95}
        ],
        [
            {'bbox': [110, 110, 210, 210], 'score': 0.95},
            {'bbox': [310, 310, 410, 410], 'score': 0.95}
        ]
    ]
    
    track_ids = set()
    for frame_dets in detections:
        tracks = tracker.update(mock_frame, frame_dets)
        for track in tracks:
            track_ids.add(int(track[4]))
    
    # Verify concurrent tracking
    assert len(track_ids) == 2  # Two distinct tracks
    for track_id in track_ids:
        assert track_id in tracker.track_embeddings
        assert len(tracker.track_embeddings[track_id]) > 0
        
        # Verify track history
        assert track_id in tracker.track_history
        history = tracker.track_history[track_id]
        assert len(history) > 0
        
        # Verify track age
        assert track_id in tracker.track_ages
        assert tracker.track_ages[track_id] > 0

def test_track_id_persistence_with_behavior(mock_frame):
    """Test track ID persistence while tracking behaviors."""
    tracker = EnhancedTracker()
    
    # Create sequence with behaviors and occlusions
    detections = [
        {'bbox': [100, 100, 200, 200], 'score': 0.95, 'behavior': 'perching'},
        {'bbox': [150, 150, 250, 250], 'score': 0.95, 'behavior': 'flying'},
        None,  # Occlusion
        {'bbox': [200, 200, 300, 300], 'score': 0.95, 'behavior': 'perching'},
        {'bbox': [250, 250, 350, 350], 'score': 0.95, 'behavior': 'flying'}
    ]
    
    track_ids = set()
    behaviors = []
    
    for det in detections:
        if det is not None:
            tracks = tracker.update(mock_frame, [det])
            if len(tracks) > 0:
                track_id = int(tracks[0][4])
                track_ids.add(track_id)
                if 'behavior' in det:
                    behaviors.append(det['behavior'])
        else:
            tracker.update(mock_frame, [])
    
    # Verify track ID persistence
    assert len(track_ids) == 1, "Track ID should persist through behaviors and occlusions"
    
    # Verify behavior tracking
    track_id = track_ids.pop()
    history = tracker.track_id_changes[track_id]['history']
    for behavior in behaviors:
        assert any(behavior in str(entry) for entry in history) 

def test_multi_view_error_handling(mock_multi_view_frame):
    """Test error handling in multi-view processing."""
    tracker = EnhancedTracker(multi_view_stride=2)
    
    # Test invalid view information
    invalid_detections = [
        {
            'bbox': [100, 100, 200, 200],
            'score': 0.95,
            'class': 'crow',
            'view': None  # Invalid view
        },
        {
            'bbox': [200, 200, 300, 300],
            'score': 0.85,
            'class': 'crow',
            'view': ''  # Empty view
        }
    ]
    
    # Should handle invalid views gracefully
    tracks = tracker.update(mock_multi_view_frame, invalid_detections)
    assert len(tracks) > 0
    
    # Test missing view information
    missing_view_detections = [
        {
            'bbox': [100, 100, 200, 200],
            'score': 0.95,
            'class': 'crow'
        }
    ]
    
    tracks = tracker.update(mock_multi_view_frame, missing_view_detections)
    assert len(tracks) > 0
    
    # Test inconsistent view information
    inconsistent_detections = [
        {
            'bbox': [100, 100, 200, 200],
            'score': 0.95,
            'class': 'crow',
            'view': 'front'
        },
        {
            'bbox': [150, 150, 250, 250],
            'score': 0.95,
            'class': 'crow',
            'view': 'invalid_view'  # Invalid view type
        }
    ]
    
    tracks = tracker.update(mock_multi_view_frame, inconsistent_detections)
    assert len(tracks) > 0

def test_memory_leak_detection(mock_frame):
    """Test for memory leaks during extended operation."""
    tracker = EnhancedTracker()
    
    # Record initial memory usage
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated()
    
    # Simulate extended operation
    memory_samples = []
    for i in range(100):  # Run for many frames
        # Create detections with varying properties
        detections = [
            {
                'bbox': [100 + i, 100 + i, 200 + i, 200 + i],
                'score': 0.95,
                'class': 'crow',
                'behavior': 'perching' if i % 2 == 0 else 'flying'
            }
            for _ in range(3)  # Multiple objects per frame
        ]
        
        # Update tracker
        tracker.update(mock_frame, detections)
        
        # Sample memory usage periodically
        if i % 10 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()
            memory_samples.append(torch.cuda.memory_allocated())
    
    # Verify memory usage
    if torch.cuda.is_available():
        # Check for memory growth
        memory_growth = [s - initial_memory for s in memory_samples]
        max_growth = max(memory_growth)
        assert max_growth < 1e8  # Less than 100MB growth
        
        # Check for memory stability
        memory_variance = np.var(memory_growth)
        assert memory_variance < 1e7  # Stable memory usage

def test_behavioral_patterns(mock_frame):
    """Test tracking of complex behavioral patterns."""
    tracker = EnhancedTracker()
    
    # Define a sequence of behaviors
    behavior_sequence = [
        ('perching', 5),    # Perch for 5 frames
        ('flying', 3),      # Fly for 3 frames
        ('perching', 2),    # Perch for 2 frames
        ('flying', 4),      # Fly for 4 frames
        ('perching', 3)     # Perch for 3 frames
    ]
    
    # Generate detections with behaviors
    detections = []
    base_bbox = [100, 100, 200, 200]
    frame_count = 0
    
    for behavior, duration in behavior_sequence:
        for i in range(duration):
            # Adjust bbox based on behavior
            if behavior == 'flying':
                offset = i * 10  # More movement for flying
            else:
                offset = i * 2   # Less movement for perching
            
            bbox = [
                base_bbox[0] + offset,
                base_bbox[1] + offset,
                base_bbox[2] + offset,
                base_bbox[3] + offset
            ]
            
            detections.append({
                'bbox': bbox,
                'score': 0.95,
                'class': 'crow',
                'behavior': behavior,
                'frame': frame_count
            })
            frame_count += 1
    
    # Process detections and verify behavior tracking
    track_ids = set()
    behavior_history = []
    
    for det in detections:
        tracks = tracker.update(mock_frame, [det])
        if len(tracks) > 0:
            track_id = int(tracks[0][4])
            track_ids.add(track_id)
            if 'behavior' in det:
                behavior_history.append(det['behavior'])
    
    # Verify behavior sequence
    assert len(track_ids) == 1, "Should maintain single track through behavior changes"
    track_id = track_ids.pop()
    
    # Verify behavior transitions
    history = tracker.track_id_changes[track_id]['history']
    behavior_transitions = []
    current_behavior = None
    
    for entry in history:
        if 'behavior' in str(entry):
            for behavior, _ in behavior_sequence:
                if behavior in str(entry):
                    if behavior != current_behavior:
                        behavior_transitions.append(behavior)
                        current_behavior = behavior
                    break
    
    # Verify behavior sequence matches expected pattern
    expected_behaviors = [b for b, _ in behavior_sequence]
    assert behavior_transitions == expected_behaviors, "Behavior sequence mismatch"

def test_track_id_reassignment(mock_frame):
    """Test track ID reassignment scenarios."""
    tracker = EnhancedTracker(max_age=5, min_hits=2, iou_threshold=0.15)
    
    # Create sequence that forces track ID reassignment
    sequences = [
        # Sequence 1: Object moves normally
        [
            {'bbox': [100, 100, 200, 200], 'score': 0.95},
            {'bbox': [110, 110, 210, 210], 'score': 0.95},
            {'bbox': [120, 120, 220, 220], 'score': 0.95}
        ],
        # Sequence 2: Object disappears and reappears
        [
            {'bbox': [300, 300, 400, 400], 'score': 0.95},
            None,  # Missing detection
            {'bbox': [320, 320, 420, 420], 'score': 0.95}
        ],
        # Sequence 3: Object moves rapidly
        [
            {'bbox': [500, 500, 600, 600], 'score': 0.95},
            {'bbox': [700, 700, 800, 800], 'score': 0.95},  # Large movement
            {'bbox': [750, 750, 850, 850], 'score': 0.95}
        ]
    ]
    
    track_id_changes = []
    
    for sequence in sequences:
        sequence_track_ids = set()
        for det in sequence:
            if det is not None:
                tracks = tracker.update(mock_frame, [det])
                if len(tracks) > 0:
                    track_id = int(tracks[0][4])
                    sequence_track_ids.add(track_id)
            else:
                tracker.update(mock_frame, [])
        
        if sequence_track_ids:
            track_id_changes.append(sequence_track_ids)
    
    # Verify track ID behavior
    for i, track_ids in enumerate(track_id_changes):
        if i == 0:  # Normal movement
            assert len(track_ids) == 1, "Should maintain track ID during normal movement"
        elif i == 1:  # Disappearance
            assert len(track_ids) <= 2, "May get new track ID after disappearance"
        elif i == 2:  # Rapid movement
            assert len(track_ids) <= 2, "May get new track ID after rapid movement"
    
    # Verify track history
    for track_ids in track_id_changes:
        for track_id in track_ids:
            assert track_id in tracker.track_id_changes
            history = tracker.track_id_changes[track_id]
            assert 'created_frame' in history
            assert 'last_seen' in history
            assert len(history['history']) > 0

def test_track_id_reassignment_with_occlusion(mock_frame):
    """Test track ID reassignment during occlusions."""
    tracker = EnhancedTracker(max_age=5, min_hits=2, iou_threshold=0.15)
    
    # Create sequence with occlusion and reappearance
    detections = [
        # Initial tracking
        {'bbox': [100, 100, 200, 200], 'score': 0.95},
        {'bbox': [110, 110, 210, 210], 'score': 0.95},
        # Occlusion
        None,
        None,
        None,
        # Reappearance
        {'bbox': [150, 150, 250, 250], 'score': 0.95},
        {'bbox': [160, 160, 260, 260], 'score': 0.95},
        # Another occlusion
        None,
        None,
        # Final reappearance
        {'bbox': [200, 200, 300, 300], 'score': 0.95},
        {'bbox': [210, 210, 310, 310], 'score': 0.95}
    ]
    
    track_ids = set()
    occlusion_periods = []
    current_period = []
    
    for i, det in enumerate(detections):
        if det is not None:
            tracks = tracker.update(mock_frame, [det])
            if len(tracks) > 0:
                track_id = int(tracks[0][4])
                track_ids.add(track_id)
                if current_period:
                    occlusion_periods.append(current_period)
                    current_period = []
        else:
            tracker.update(mock_frame, [])
            current_period.append(i)
    
    if current_period:
        occlusion_periods.append(current_period)
    
    # Verify track ID behavior
    assert len(track_ids) <= 3, "Should have at most 3 different track IDs"
    
    # Verify occlusion handling
    for period in occlusion_periods:
        assert len(period) <= tracker.max_age, "Occlusion period should not exceed max_age"
    
    # Verify track history
    for track_id in track_ids:
        history = tracker.track_id_changes[track_id]
        assert 'created_frame' in history
        assert 'last_seen' in history
        assert any('occlusion' in str(entry) for entry in history['history']) 

def test_device_synchronization():
    """Test that device operations are properly synchronized."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
        
    tracker = EnhancedTracker()
    
    # Test device context manager
    with tracker.device_context('cuda'):
        assert tracker.device.type == 'cuda'
        # Verify synchronization
        assert torch.cuda.is_initialized()
        torch.cuda.synchronize()
        
    # Test device transitions
    with tracker.device_context('cpu'):
        assert tracker.device.type == 'cpu'
        with tracker.device_context('cuda'):
            assert tracker.device.type == 'cuda'
            torch.cuda.synchronize()
        assert tracker.device.type == 'cpu'

def test_temporal_consistency_tracking(mock_frame):
    """Test that temporal consistency is properly tracked."""
    tracker = EnhancedTracker()
    
    # Create detections with temporal consistency
    detections = [{
        'bbox': [100, 100, 200, 200],
        'score': 0.9,
        'class': 'crow',
        'temporal_consistency': 0.8
    }]
    
    # Update tracker
    tracks = tracker.update(mock_frame, detections)
    assert len(tracks) > 0
    
    # Verify temporal consistency is tracked
    track_id = int(tracks[0][4])
    assert track_id in tracker.track_history
    assert 'temporal_consistency' in tracker.track_history[track_id]
    assert tracker.track_history[track_id]['temporal_consistency'] > 0

def test_track_history_management(mock_frame):
    """Test that track history is properly managed with deque limits."""
    tracker = EnhancedTracker()
    
    # Create detections
    detections = [{
        'bbox': [100, 100, 200, 200],
        'score': 0.9,
        'class': 'crow'
    }]
    
    # Update tracker multiple times
    for _ in range(150):  # More than deque maxlen
        tracks = tracker.update(mock_frame, detections)
        if len(tracks) > 0:
            track_id = int(tracks[0][4])
            # Verify history size is limited
            assert len(tracker.track_history[track_id]['history']) <= 100
            assert len(tracker.track_embeddings[track_id]) <= 100
            assert len(tracker.track_head_embeddings[track_id]) <= 100

def test_embedding_processing_amp(mock_frame):
    """Test embedding processing with automatic mixed precision."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
        
    tracker = EnhancedTracker()
    
    # Create detections
    detections = [{
        'bbox': [100, 100, 200, 200],
        'score': 0.9,
        'class': 'crow'
    }]
    
    # Update tracker with AMP enabled
    with torch.cuda.amp.autocast():
        tracks = tracker.update(mock_frame, detections)
        assert len(tracks) > 0
        
        # Verify embeddings were processed
        track_id = int(tracks[0][4])
        assert track_id in tracker.track_embeddings
        assert len(tracker.track_embeddings[track_id]) > 0
        assert len(tracker.track_head_embeddings[track_id]) > 0

def test_memory_management_deque(mock_frame):
    """Test that memory is properly managed with deque limits."""
    tracker = EnhancedTracker()
    
    # Create detections
    detections = [{
        'bbox': [100, 100, 200, 200],
        'score': 0.9,
        'class': 'crow'
    }]
    
    # Track memory usage
    import psutil
    process = psutil.Process()
    initial_memory = process.memory_info().rss
    
    # Update tracker many times
    for _ in range(1000):
        tracks = tracker.update(mock_frame, detections)
        if len(tracks) > 0:
            track_id = int(tracks[0][4])
            # Verify deque limits are enforced
            assert len(tracker.track_embeddings[track_id]) <= 100
            assert len(tracker.track_head_embeddings[track_id]) <= 100
            assert len(tracker.track_history[track_id]['history']) <= 100
            assert len(tracker.track_history[track_id]['behaviors']) <= 50
    
    # Verify memory usage hasn't grown unbounded
    final_memory = process.memory_info().rss
    memory_growth = final_memory - initial_memory
    assert memory_growth < 100 * 1024 * 1024  # Less than 100MB growth

def test_track_id_persistence_with_temporal_consistency(mock_frame):
    """Test that track IDs persist better with temporal consistency."""
    tracker = EnhancedTracker()
    
    # Create detections with varying temporal consistency
    detections = []
    for i in range(10):
        consistency = 0.8 if i % 2 == 0 else 0.3  # Alternate high and low consistency
        detections.append({
            'bbox': [100, 100, 200, 200],
            'score': 0.9,
            'class': 'crow',
            'temporal_consistency': consistency
        })
    
    # Track IDs should be more stable for high consistency detections
    track_ids = set()
    for det in detections:
        tracks = tracker.update(mock_frame, [det])
        if len(tracks) > 0:
            track_ids.add(int(tracks[0][4]))
    
    # Should have fewer track IDs with temporal consistency
    assert len(track_ids) < len(detections) // 2 