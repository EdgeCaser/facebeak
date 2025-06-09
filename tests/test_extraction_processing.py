import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock, call
import shutil
import json

# Add project root to sys.path
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utilities.extract_training_data import extract_crops_from_video_task # Assuming path
# from detection import detect_crows_parallel # To be mocked
# from crow_tracking import CrowTracker # Implicitly tested via extract_crops_from_video_task
# from db import save_crow_embedding # To be mocked

@pytest.fixture
def temp_extraction_base_dir(tmp_path_factory):
    # Create a temporary base directory for CrowTracker outputs
    base_dir = tmp_path_factory.mktemp("temp_extraction_data_")
    # utilities.extract_training_data.extract_crops_from_video_task will create metadata and videos dirs if needed
    return base_dir

# Define a dummy video path (doesn't need to exist for this test due to mocking)
DUMMY_VIDEO_PATH = "dummy_video.mp4"

@patch('utilities.extract_training_data.detect_crows_parallel')
@patch('crow_tracking.save_crow_embedding') # Patch where save_crow_embedding is called from (within CrowTracker)
@patch('cv2.VideoCapture') # Mock VideoCapture to avoid needing a real video file
def test_multiple_detections_single_frame_processing(
    mock_videocapture, mock_save_embedding, mock_detect_crows, temp_extraction_base_dir
):
    """
    Tests that multiple detections in a single frame result in multiple crops,
    metadata entries, and embedding save calls.
    """
    # Configure mock VideoCapture
    mock_cap_instance = MagicMock()
    mock_cap_instance.isOpened.return_value = True
    mock_cap_instance.get.side_effect = lambda prop: {
        cv2.CAP_PROP_FRAME_COUNT: 1, # Simulate a single frame video for simplicity
        cv2.CAP_PROP_FPS: 30
    }.get(prop, 0)

    # Simulate reading one frame successfully, then no more frames
    # Frame content can be minimal as detection is mocked.
    dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    read_effects = [(True, dummy_frame), (False, None)]
    mock_cap_instance.read.side_effect = read_effects
    mock_videocapture.return_value = mock_cap_instance

    # Configure mock detect_crows_parallel
    # Simulate two detections in the single frame
    detection1 = {'bbox': [10, 10, 50, 50], 'score': 0.9, 'class': 'crow'}
    detection2 = {'bbox': [60, 60, 100, 100], 'score': 0.8, 'class': 'crow'}
    # detect_crows_parallel expects a list of frames, and returns a list of detections for each frame
    mock_detect_crows.return_value = [[detection1, detection2]]

    # Call the task function
    summary = extract_crops_from_video_task(
        video_path_str=DUMMY_VIDEO_PATH,
        base_dir_str=str(temp_extraction_base_dir),
        min_confidence_val=0.1,
        frame_skip_val=0,
        batch_size_frames_val=1, # Process one frame at a time
        enable_audio_val=False,
        correct_orientation_val=False,
        target_frame_rate_val=None
    )

    assert summary["status"] == "success"
    assert summary["detections_count_in_video"] == 2
    # Crow IDs identified depends on CrowTracker logic, check at least one was processed.
    # This might be 1 or 2 depending on if they are considered the same crow by tracker in this simple case.
    # For this test, we focus on crop creation and embedding saving per detection.
    # The number of *unique* crow IDs might be 1 if tracker merges these close detections,
    # but two *crops* should still be saved if they are distinct enough after normalization,
    # and two *embeddings* should be saved to DB.

    video_name = Path(DUMMY_VIDEO_PATH).stem
    video_crops_dir = temp_extraction_base_dir / "videos" / video_name

    # Assert that two crop files were created (one for each detection)
    # CrowTracker's save_crop ensures unique filenames even for same frame.
    # Example: frame_000000_crop_001.jpg, frame_000000_crop_002.jpg
    created_crop_files = list(video_crops_dir.glob("frame_000000_crop_*.jpg"))
    assert len(created_crop_files) == 2, "Should create one crop file per detection"

    # Assert that crop_metadata.json was created and contains entries for these crops
    metadata_file = temp_extraction_base_dir / "metadata" / "crop_metadata.json"
    assert metadata_file.exists(), "crop_metadata.json should be created"

    with open(metadata_file, 'r') as f:
        crop_metadata = json.load(f)

    assert len(crop_metadata["crops"]) == 2, "crop_metadata.json should have two entries"
    # Check if paths in metadata match created files (relative to base_dir)
    relative_crop_paths_in_metadata = set(crop_metadata["crops"].keys())
    expected_relative_paths = {
        f"videos/{video_name}/frame_000000_crop_001.jpg",
        f"videos/{video_name}/frame_000000_crop_002.jpg"
    }
    # This assertion might be too strict if filename generation is slightly different,
    # better to check count and that they belong to the right frame.
    # For now, assuming this naming.
    # assert relative_crop_paths_in_metadata == expected_relative_paths

    # Assert that db.save_crow_embedding was called twice (once for each processed detection)
    # CrowTracker.process_detection calls save_crow_embedding
    assert mock_save_embedding.call_count == 2, "save_crow_embedding should be called for each valid detection"

    # Further checks on mock_save_embedding arguments (optional, more involved)
    # For example, check that embeddings passed are numpy arrays, video_path matches, etc.
    args_list = mock_save_embedding.call_args_list
    assert args_list[0][1]['video_path'] == DUMMY_VIDEO_PATH
    assert args_list[0][1]['frame_number'] == 0
    assert args_list[0][1]['confidence'] == detection1['score']

    assert args_list[1][1]['video_path'] == DUMMY_VIDEO_PATH
    assert args_list[1][1]['frame_number'] == 0
    assert args_list[1][1]['confidence'] == detection2['score']
