import pytest
import os
import shutil
import numpy as np
import torch
from pathlib import Path
from datetime import datetime, timedelta
from crow_tracking import CrowTracker
from tracking import extract_crow_image

@pytest.fixture
def temp_tracker_dir(tmp_path):
    """Create a temporary directory for tracker data."""
    tracker_dir = tmp_path / "crow_tracker_test"
    tracker_dir.mkdir()
    yield tracker_dir
    # Cleanup after tests
    if tracker_dir.exists():
        shutil.rmtree(tracker_dir)

@pytest.fixture
def mock_detection():
    """Create a mock detection for testing."""
    return {
        'bbox': np.array([100, 100, 200, 200]),  # x1, y1, x2, y2
        'score': 0.95,
        'class': 'crow'
    }

@pytest.fixture
def mock_frame():
    """Create a mock frame for testing."""
    # Create a 400x400 RGB image with a white rectangle
    frame = np.zeros((400, 400, 3), dtype=np.uint8)
    frame[100:200, 100:200] = [255, 255, 255]  # White rectangle
    return frame

@pytest.fixture
def tracker(temp_tracker_dir):
    """Create a CrowTracker instance for testing."""
    tracker = CrowTracker(base_dir=str(temp_tracker_dir))
    return tracker

def test_tracker_initialization(temp_tracker_dir):
    """Test tracker initialization and directory creation."""
    tracker = CrowTracker(base_dir=str(temp_tracker_dir))
    
    # Check if directories were created
    assert (temp_tracker_dir / "crows").exists()
    assert (temp_tracker_dir / "processing").exists()
    assert (temp_tracker_dir / "metadata").exists()
    assert (temp_tracker_dir / "metadata" / "crow_tracking.json").exists()
    
    # Check initial tracking data
    assert tracker.tracking_data["crows"] == {}
    assert tracker.tracking_data["last_id"] == 0
    assert "created_at" in tracker.tracking_data
    assert "updated_at" in tracker.tracking_data

def test_load_existing_tracking_data(temp_tracker_dir):
    """Test loading existing tracking data."""
    # Create initial tracker
    tracker1 = CrowTracker(base_dir=str(temp_tracker_dir))
    tracker1.tracking_data["last_id"] = 5
    tracker1._save_tracking_data(force=True)
    
    # Create new tracker instance and verify data was loaded
    tracker2 = CrowTracker(base_dir=str(temp_tracker_dir))
    assert tracker2.tracking_data["last_id"] == 5

def test_generate_crow_id(tracker):
    """Test crow ID generation."""
    # Generate first ID
    crow_id1 = tracker._generate_crow_id()
    assert crow_id1 == "crow_0001"
    assert tracker.tracking_data["last_id"] == 1
    
    # Generate second ID
    crow_id2 = tracker._generate_crow_id()
    assert crow_id2 == "crow_0002"
    assert tracker.tracking_data["last_id"] == 2

def test_process_detection_new_crow(tracker, mock_frame, mock_detection):
    """Test processing a detection for a new crow."""
    frame_num = 1
    video_path = "test_video.mp4"
    frame_time = datetime.now()
    
    # Process detection
    crow_id = tracker.process_detection(mock_frame, frame_num, mock_detection, video_path, frame_time)
    
    # Verify crow was created
    assert crow_id is not None
    assert crow_id in tracker.tracking_data["crows"]
    
    # Verify crow data
    crow_data = tracker.tracking_data["crows"][crow_id]
    assert crow_data["total_detections"] == 1
    assert crow_data["first_frame"] == frame_num
    assert crow_data["last_frame"] == frame_num
    assert crow_data["video_path"] == video_path
    assert len(crow_data["detections"]) == 1
    
    # Verify detection record
    detection = crow_data["detections"][0]
    assert detection["frame"] == frame_num
    assert np.array_equal(detection["bbox"], mock_detection["bbox"].tolist())
    assert detection["score"] == mock_detection["score"]
    
    # Verify crop was saved
    crow_dir = tracker.crows_dir / crow_id
    assert crow_dir.exists()
    assert len(list(crow_dir.glob("*.jpg"))) == 1

def test_process_detection_existing_crow(tracker, mock_frame, mock_detection):
    """Test processing a detection for an existing crow."""
    frame_num1 = 1
    frame_num2 = 2
    video_path = "test_video.mp4"
    frame_time = datetime.now()
    
    # Process first detection
    crow_id = tracker.process_detection(mock_frame, frame_num1, mock_detection, video_path, frame_time)
    assert crow_id is not None
    
    # Process second detection
    crow_id2 = tracker.process_detection(mock_frame, frame_num2, mock_detection, video_path, frame_time)
    
    # Verify same crow was updated
    assert crow_id2 == crow_id
    crow_data = tracker.tracking_data["crows"][crow_id]
    assert crow_data["total_detections"] == 2
    assert crow_data["last_frame"] == frame_num2
    assert len(crow_data["detections"]) == 2

def test_find_matching_crow(tracker, mock_frame, mock_detection):
    """Test finding a matching crow using embeddings."""
    frame_num = 1
    video_path = "test_video.mp4"
    frame_time = datetime.now()
    
    # Process first detection
    crow_id1 = tracker.process_detection(mock_frame, frame_num, mock_detection, video_path, frame_time)
    assert crow_id1 is not None
    
    # Process similar detection
    similar_detection = mock_detection.copy()
    similar_detection['bbox'] = mock_detection['bbox'] + np.array([10, 10, 10, 10])  # Slightly offset
    crow_id2 = tracker.process_detection(mock_frame, frame_num + 1, similar_detection, video_path, frame_time)
    
    # Should match the first crow
    assert crow_id2 == crow_id1

def test_get_crow_info(tracker, mock_frame, mock_detection):
    """Test retrieving crow information."""
    frame_num = 1
    video_path = "test_video.mp4"
    frame_time = datetime.now()
    
    # Process detection
    crow_id = tracker.process_detection(mock_frame, frame_num, mock_detection, video_path, frame_time)
    
    # Get crow info
    info = tracker.get_crow_info(crow_id)
    assert info is not None
    assert info["total_detections"] == 1
    assert info["first_frame"] == frame_num
    assert info["video_path"] == video_path
    
    # Test non-existent crow
    assert tracker.get_crow_info("nonexistent_crow") is None

def test_list_crows(tracker, mock_frame, mock_detection):
    """Test listing all crows."""
    frame_num = 1
    video_path = "test_video.mp4"
    frame_time = datetime.now()
    
    # Create two distinct frames
    frame1 = mock_frame.copy()
    frame2 = mock_frame.copy()
    frame2[150:250, 150:250] = [0, 255, 0]  # Add a green square to make it different
    
    # Process two detections on different frames
    crow_id1 = tracker.process_detection(frame1, frame_num, mock_detection, video_path, frame_time)
    crow_id2 = tracker.process_detection(frame2, frame_num + 1, mock_detection, video_path, frame_time)
    
    # List crows
    crows = tracker.list_crows()
    assert len(crows) == 2
    assert crow_id1 in crows
    assert crow_id2 in crows
    
    # Verify crow info
    for crow_id in [crow_id1, crow_id2]:
        assert crows[crow_id]["total_detections"] == 1
        assert crows[crow_id]["video_path"] == video_path

def test_processing_run_management(tracker):
    """Test processing run directory management."""
    # Create processing run
    run_dir = tracker.create_processing_run()
    assert run_dir.exists()
    assert run_dir.is_dir()
    
    # Cleanup processing run
    tracker.cleanup_processing_dir(run_dir)
    assert not run_dir.exists()

def test_invalid_detection_handling(tracker, mock_frame):
    """Test handling of invalid detections."""
    # Test with invalid box coordinates
    invalid_detection = {
        'bbox': np.array([-100, -100, 0, 0]),  # Invalid coordinates
        'score': 0.95,
        'class': 'crow'
    }
    
    crow_id = tracker.process_detection(mock_frame, 1, invalid_detection, "test.mp4", datetime.now())
    assert crow_id is None
    
    # Test with invalid score
    invalid_detection = {
        'bbox': np.array([100, 100, 200, 200]),
        'score': -0.5,  # Invalid score
        'class': 'crow'
    }
    
    crow_id = tracker.process_detection(mock_frame, 1, invalid_detection, "test.mp4", datetime.now())
    assert crow_id is None

def test_save_tracking_data(tracker, mock_frame, mock_detection):
    """Test saving tracking data."""
    # Process a detection
    frame_num = 1
    video_path = "test_video.mp4"
    frame_time = datetime.now()
    crow_id = tracker.process_detection(mock_frame, frame_num, mock_detection, video_path, frame_time)
    
    # Force save
    tracker._save_tracking_data(force=True)
    
    # Verify file exists and contains correct data
    tracking_file = tracker.tracking_file
    assert tracking_file.exists()
    
    # Create new tracker instance to verify data was saved correctly
    new_tracker = CrowTracker(base_dir=str(tracker.base_dir))
    assert new_tracker.tracking_data["crows"][crow_id]["total_detections"] == 1
    assert new_tracker.tracking_data["last_id"] == tracker.tracking_data["last_id"] 