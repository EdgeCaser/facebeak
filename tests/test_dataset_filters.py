import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
import random
import shutil

# Add project root to sys.path to allow importing project modules
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from improved_dataset import ImprovedCrowTripletDataset # Assuming this is the correct import path
# from db import get_image_label # This will be mocked

# Define a constant for labels used in tests
LABEL_CROW = 'crow'
LABEL_NOT_A_CROW = 'not_a_crow'
LABEL_MULTI_CROW = 'multi_crow'
LABEL_BAD_CROW = 'bad_crow'

@pytest.fixture
def temp_dataset_base_dir(tmp_path_factory):
    # Create a temporary base directory structure similar to crow_crops
    base_dir = tmp_path_factory.mktemp("temp_base_data_")

    # Create metadata directory
    metadata_dir = base_dir / "metadata"
    metadata_dir.mkdir()

    # Create videos directory
    videos_dir = base_dir / "videos"
    videos_dir.mkdir()

    # Create a dummy video sub-directory
    video1_dir = videos_dir / "test_video1"
    video1_dir.mkdir()

    # Create some dummy image files
    # These are just placeholders; their content doesn't matter for these tests
    # as _load_image is not the focus here, but filtering based on metadata and DB labels.
    (video1_dir / "frame_000001_crop_001.jpg").touch()
    (video1_dir / "frame_000002_crop_001.jpg").touch()
    (video1_dir / "frame_000003_crop_001.jpg").touch()
    (video1_dir / "frame_000004_crop_001.jpg").touch()
    (video1_dir / "frame_000005_crop_001.jpg").touch()

    # Create dummy crop_metadata.json
    crop_metadata_content = {
        "crops": {
            "videos/test_video1/frame_000001_crop_001.jpg": {"crow_id": "crow_A", "frame": 1, "video": "test_video1"},
            "videos/test_video1/frame_000002_crop_001.jpg": {"crow_id": "crow_A", "frame": 2, "video": "test_video1"}, # multi_crow test
            "videos/test_video1/frame_000003_crop_001.jpg": {"crow_id": "crow_B", "frame": 3, "video": "test_video1"}, # not_a_crow test
            "videos/test_video1/frame_000004_crop_001.jpg": {"crow_id": "crow_C", "frame": 4, "video": "test_video1"}, # not training data test
            "videos/test_video1/frame_000005_crop_001.jpg": {"crow_id": "crow_D", "frame": 5, "video": "test_video1"}  # valid crow
        },
        "created_at": "2023-01-01T00:00:00",
        "updated_at": "2023-01-01T00:00:00"
    }
    with open(metadata_dir / "crop_metadata.json", 'w') as f:
        json.dump(crop_metadata_content, f)

    return base_dir

# Mock for db.get_image_label
# This can be a dictionary mapping image paths (relative to base_dir) to their label info
MOCKED_DB_LABELS = {
    "videos/test_video1/frame_000001_crop_001.jpg": {"label": LABEL_CROW, "is_training_data": True},
    "videos/test_video1/frame_000002_crop_001.jpg": {"label": LABEL_MULTI_CROW, "is_training_data": True},
    "videos/test_video1/frame_000003_crop_001.jpg": {"label": LABEL_NOT_A_CROW, "is_training_data": True},
    "videos/test_video1/frame_000004_crop_001.jpg": {"label": LABEL_CROW, "is_training_data": False},
    "videos/test_video1/frame_000005_crop_001.jpg": {"label": LABEL_CROW, "is_training_data": True}, # Valid
}

@patch('improved_dataset.get_image_label') # Patch where it's looked up (module where ImprovedCrowTripletDataset is)
def test_exclude_multi_crow(mock_get_image_label, temp_dataset_base_dir):
    """Test that an image labeled 'multi_crow' is excluded."""

    def side_effect_get_image_label(image_path_str):
        # image_path_str will be absolute, convert to relative for lookup in MOCKED_DB_LABELS
        relative_path = str(Path(image_path_str).relative_to(temp_dataset_base_dir))
        return MOCKED_DB_LABELS.get(relative_path)

    mock_get_image_label.side_effect = side_effect_get_image_label

    # Initialize dataset for 'train' split (assuming all crow_ids A,B,C,D are part of train by default for small dataset)
    # To make train/val split more robust in tests, we might need more crow_ids or to mock train_test_split
    # For now, let's assume crow_A, B, C, D are split such that A is in train.
    # Let's adjust metadata to ensure crow_A is in train for this specific test.
    # A simpler way for this unit test is to make all IDs go to train split to check filtering.
    # We can test split separately.
    with patch('improved_dataset.train_test_split', return_value=(['crow_A', 'crow_B', 'crow_C', 'crow_D'], [])): # All to train
        dataset = ImprovedCrowTripletDataset(
            base_dir=str(temp_dataset_base_dir),
            split='train', # or 'val' depending on what you want to test for the split
            min_samples_per_crow=1 # Ensure crows are not dropped due to too few samples after filtering
        )

    # Check that the image labeled 'multi_crow' (frame_000002_crop_001.jpg, crow_A) is not in the dataset samples
    multi_crow_image_path = temp_dataset_base_dir / "videos/test_video1/frame_000002_crop_001.jpg"

    found_in_samples = any(sample_path == multi_crow_image_path for sample_path, _ in dataset.all_image_paths_labels)
    assert not found_in_samples, "Multi-crow image should be excluded from all_image_paths_labels"

    assert multi_crow_image_path not in dataset.crow_to_imgs.get('crow_A', []), "Multi-crow image should be excluded from crow_to_imgs"

    # Check that a valid image for crow_A (frame_000001) IS present
    valid_crow_A_image_path = temp_dataset_base_dir / "videos/test_video1/frame_000001_crop_001.jpg"
    assert valid_crow_A_image_path in dataset.crow_to_imgs.get('crow_A', []), "Valid image for crow_A should be present"

@patch('improved_dataset.get_image_label')
def test_exclude_not_a_crow(mock_get_image_label, temp_dataset_base_dir):
    """Test that an image labeled 'not_a_crow' is excluded."""
    def side_effect(image_path_str):
        relative_path = str(Path(image_path_str).relative_to(temp_dataset_base_dir))
        return MOCKED_DB_LABELS.get(relative_path)
    mock_get_image_label.side_effect = side_effect

    with patch('improved_dataset.train_test_split', return_value=(['crow_A', 'crow_B', 'crow_C', 'crow_D'], [])):
        dataset = ImprovedCrowTripletDataset(str(temp_dataset_base_dir), split='train', min_samples_per_crow=1)

    not_a_crow_image_path = temp_dataset_base_dir / "videos/test_video1/frame_000003_crop_001.jpg" # Belongs to crow_B

    found_in_samples = any(p == not_a_crow_image_path for p, _ in dataset.all_image_paths_labels)
    assert not found_in_samples, "'not_a_crow' image should be excluded"
    assert not_a_crow_image_path not in dataset.crow_to_imgs.get('crow_B', [])


@patch('improved_dataset.get_image_label')
def test_exclude_not_training_data(mock_get_image_label, temp_dataset_base_dir):
    """Test that an image with is_training_data=False is excluded."""
    def side_effect(image_path_str):
        relative_path = str(Path(image_path_str).relative_to(temp_dataset_base_dir))
        return MOCKED_DB_LABELS.get(relative_path)
    mock_get_image_label.side_effect = side_effect

    with patch('improved_dataset.train_test_split', return_value=(['crow_A', 'crow_B', 'crow_C', 'crow_D'], [])):
        dataset = ImprovedCrowTripletDataset(str(temp_dataset_base_dir), split='train', min_samples_per_crow=1)

    not_training_image_path = temp_dataset_base_dir / "videos/test_video1/frame_000004_crop_001.jpg" # Belongs to crow_C

    found_in_samples = any(p == not_training_image_path for p, _ in dataset.all_image_paths_labels)
    assert not found_in_samples, "is_training_data=False image should be excluded"
    assert not_training_image_path not in dataset.crow_to_imgs.get('crow_C', [])


@patch('improved_dataset.get_image_label')
def test_include_valid_crow_image(mock_get_image_label, temp_dataset_base_dir):
    """Test that a valid crow image is included."""
    def side_effect(image_path_str):
        relative_path = str(Path(image_path_str).relative_to(temp_dataset_base_dir))
        return MOCKED_DB_LABELS.get(relative_path)
    mock_get_image_label.side_effect = side_effect

    with patch('improved_dataset.train_test_split', return_value=(['crow_A', 'crow_B', 'crow_C', 'crow_D'], [])): # All to train
        dataset = ImprovedCrowTripletDataset(str(temp_dataset_base_dir), split='train', min_samples_per_crow=1)

    valid_image_path = temp_dataset_base_dir / "videos/test_video1/frame_000005_crop_001.jpg" # Belongs to crow_D

    found_in_samples = any(p == valid_image_path for p, l in dataset.all_image_paths_labels if l == 'crow_D')
    assert found_in_samples, "Valid image should be included in all_image_paths_labels"
    assert valid_image_path in dataset.crow_to_imgs.get('crow_D', []), "Valid image should be in crow_to_imgs for crow_D"

@patch('improved_dataset.get_image_label') # Mock get_image_label even if not directly used by split logic, good practice
def test_train_val_split_ids(mock_get_image_label, temp_dataset_base_dir):
    """Test that train and val splits get different, non-overlapping crow_ids."""
    # For this test, make all DB labels valid so filtering doesn't interfere with checking split logic
    def side_effect_get_image_label(image_path_str):
        return {"label": LABEL_CROW, "is_training_data": True} # All images are valid crows
    mock_get_image_label.side_effect = side_effect_get_image_label

    # Modify metadata for more robust split testing
    metadata_dir = temp_dataset_base_dir / "metadata"
    more_crows_metadata = {
        "crops": {
            "videos/test_video1/frame_000001_crop_001.jpg": {"crow_id": "crow_A", "frame": 1, "video": "test_video1"},
            "videos/test_video1/frame_000002_crop_001.jpg": {"crow_id": "crow_B", "frame": 2, "video": "test_video1"},
            "videos/test_video1/frame_000003_crop_001.jpg": {"crow_id": "crow_C", "frame": 3, "video": "test_video1"},
            "videos/test_video1/frame_000004_crop_001.jpg": {"crow_id": "crow_D", "frame": 4, "video": "test_video1"},
            "videos/test_video1/frame_000005_crop_001.jpg": {"crow_id": "crow_E", "frame": 5, "video": "test_video1"}
        }
    }
    with open(metadata_dir / "crop_metadata.json", 'w') as f:
        json.dump(more_crows_metadata, f)

    # Actual train_test_split will be called by the dataset loader.
    # We expect roughly 80/20 split. For 5 items, it's 4 train, 1 val with random_state=42.
    # The exact IDs depend on the sorted list of unique crow IDs: ['crow_A', 'crow_B', 'crow_C', 'crow_D', 'crow_E']
    # With random_state=42, train_test_split(['A','B','C','D','E'], test_size=0.2, random_state=42)
    # would result in (for example) train_ids=['A','B','D','E'], val_ids=['C'] or similar.
    # We need to ensure `sklearn.model_selection.train_test_split` is available in the test environment
    # or mock its behavior if we want to assert specific IDs.
    # For this test, we'll check for non-overlap and presence of some IDs.

    dataset_train = ImprovedCrowTripletDataset(
        base_dir=str(temp_dataset_base_dir),
        split='train',
        min_samples_per_crow=1
    )
    dataset_val = ImprovedCrowTripletDataset(
        base_dir=str(temp_dataset_base_dir),
        split='val',
        min_samples_per_crow=1
    )

    train_ids = set(dataset_train.crow_to_imgs.keys())
    val_ids = set(dataset_val.crow_to_imgs.keys())

    assert len(train_ids) > 0, "Train set should have some crow IDs"
    assert len(val_ids) > 0, "Validation set should have some crow IDs"
    assert train_ids.isdisjoint(val_ids), "Train and validation crow_ids should be disjoint"

    all_expected_ids = {'crow_A', 'crow_B', 'crow_C', 'crow_D', 'crow_E'}
    assert (train_ids | val_ids) == all_expected_ids, "Union of train and val IDs should be all unique IDs from metadata"

    # Check typical 80/20 split for 5 items (4 train, 1 val)
    # This depends on the stability of train_test_split's random_state behavior across versions/platforms
    # which is generally good but can be a source of flaky tests if not handled.
    # For this example, let's assume it splits 4/1.
    assert len(train_ids) == 4 # Based on 80% of 5, with default rounding of train_test_split
    assert len(val_ids) == 1
