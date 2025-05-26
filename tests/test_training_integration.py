#!/usr/bin/env python3
"""
Test manual labeling integration with training pipeline.
Tests the "innocent until proven guilty" philosophy and filtering logic.
"""

import unittest
from unittest.mock import patch, MagicMock, call
import tempfile
import os
import numpy as np
from pathlib import Path
import sqlite3
import shutil

# Import modules to test
from db import (
    initialize_database,
    add_image_label,
    get_image_label,
    get_training_data_stats,
    get_unlabeled_images
)


class TestTrainingIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures shared across all tests."""
        # Create temporary directory for test data
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.db_path = os.path.join(cls.temp_dir.name, "test_training_integration.db")
        cls.crop_dir = os.path.join(cls.temp_dir.name, "test_crow_crops")
        
        # Set environment variable for test database
        os.environ['CROW_DB_PATH'] = cls.db_path
        
        # Create test crop directory structure
        os.makedirs(cls.crop_dir, exist_ok=True)
        
    def setUp(self):
        """Set up test fixtures for each test."""
        # Initialize fresh database
        initialize_database()
        
        # Clear any existing image labels from previous tests
        from db import get_connection
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM image_labels")
        conn.commit()
        conn.close()
        
        # Create test images in crop directory
        self.test_images = []
        for i in range(5):
            crow_dir = os.path.join(self.crop_dir, f"crow_{i}")
            os.makedirs(crow_dir, exist_ok=True)
            img_path = os.path.join(crow_dir, f"test_image_{i}.jpg")
            Path(img_path).touch()  # Create empty file
            self.test_images.append(img_path)
            
    def tearDown(self):
        """Clean up after each test."""
        # Clear database labels
        from db import get_connection
        try:
            conn = get_connection()
            cursor = conn.cursor()
            cursor.execute("DELETE FROM image_labels")
            conn.commit()
            conn.close()
        except Exception:
            pass  # Ignore cleanup errors
            
        # Remove test images and directories
        if os.path.exists(self.crop_dir):
            shutil.rmtree(self.crop_dir)
        # Recreate crop dir for next test
        os.makedirs(self.crop_dir, exist_ok=True)
        
    @classmethod 
    def tearDownClass(cls):
        """Clean up test fixtures."""
        cls.temp_dir.cleanup()
        
    def test_crowtripleddataset_filters_not_a_crow(self):
        """Test that CrowTripletDataset excludes images labeled as 'not_a_crow'."""
        # Label some images
        add_image_label(self.test_images[0], "crow", confidence=0.95)
        add_image_label(self.test_images[1], "not_a_crow", confidence=0.90)
        add_image_label(self.test_images[2], "not_sure", confidence=0.80)
        # Leave test_images[3] and test_images[4] unlabeled
        
        # Mock the CrowTripletDataset to test filtering
        try:
            from old_scripts.train_triplet_resnet import CrowTripletDataset
            dataset = CrowTripletDataset(self.crop_dir)
            
            # Get all image paths in dataset
            dataset_paths = [str(sample[0]) for sample in dataset.samples]
            
            # Check which images are included/excluded
            included_basenames = [os.path.basename(path) for path in dataset_paths]
            
            # Labeled as crow - should be included
            self.assertIn("test_image_0.jpg", included_basenames)
            
            # Labeled as not_a_crow - should be excluded
            self.assertNotIn("test_image_1.jpg", included_basenames)
            
            # Labeled as not_sure - should be included (benefit of doubt)
            self.assertIn("test_image_2.jpg", included_basenames)
            
            # Unlabeled - should be included (innocent until proven guilty)
            self.assertIn("test_image_3.jpg", included_basenames)
            self.assertIn("test_image_4.jpg", included_basenames)
            
        except ImportError:
            self.skipTest("CrowTripletDataset not available for testing")
            
    def test_training_dataset_statistics_logging(self):
        """Test that training statistics properly reflect manual labeling."""
        # Create isolated test directory
        test_dir = os.path.join(self.temp_dir.name, "test_stats_logging")
        os.makedirs(test_dir, exist_ok=True)
        
        # Create a more realistic labeling scenario with actual files
        test_labels = [
            ("img1.jpg", "crow", 0.95),
            ("img2.jpg", "crow", 0.92),
            ("img3.jpg", "crow", 0.88),
            ("img4.jpg", "not_a_crow", 0.85),
            ("img5.jpg", "not_a_crow", 0.90),
            ("img6.jpg", "not_sure", 0.75),
            ("img7.jpg", "not_sure", 0.80)
        ]
        
        # Create test files and apply labels in our isolated directory
        test_paths = []
        for img_name, label, confidence in test_labels:
            img_path = os.path.join(test_dir, img_name)
            Path(img_path).touch()
            test_paths.append(img_path)
            add_image_label(img_path, label, confidence=confidence)
        
        try:
            # Get training statistics from our test directory
            stats = get_training_data_stats(from_directory=test_dir)
            
            # Verify counts match expected values
            self.assertEqual(stats['crow']['count'], 3)
            self.assertEqual(stats['not_a_crow']['count'], 2)  
            self.assertEqual(stats['not_sure']['count'], 2)
            self.assertEqual(stats['total_labeled'], 7)
            self.assertEqual(stats['total_excluded'], 2)  # Only not_a_crow should be excluded
        finally:
            # Clean up test files
            for test_path in test_paths:
                if os.path.exists(test_path):
                    os.remove(test_path)
            if os.path.exists(test_dir):
                shutil.rmtree(test_dir)
        
    def test_innocent_until_proven_guilty_filtering(self):
        """Test the core philosophy: include unless explicitly marked as not_a_crow."""
        # Create isolated test directory
        test_dir = os.path.join(self.temp_dir.name, "test_philosophy")
        os.makedirs(test_dir, exist_ok=True)
        
        test_scenarios = [
            # (label, should_be_included_in_training)
            (None, True),           # Unlabeled - innocent until proven guilty
            ("crow", True),         # Confirmed crow - definitely include
            ("not_sure", True),     # Uncertain - benefit of doubt
            ("not_a_crow", False),  # Proven false positive - exclude
        ]
        
        test_files = []
        try:
            for i, (label, should_include) in enumerate(test_scenarios):
                img_path = os.path.join(test_dir, f"test_scenario_{i}.jpg")
                test_files.append(img_path)
                
                # Create test file
                Path(img_path).touch()
                
                if label is not None:
                    add_image_label(img_path, label, confidence=0.8)
                
                # Check database training flag
                label_info = get_image_label(img_path)
                
                if label_info:  # Image has been labeled
                    expected_training_status = (label != "not_a_crow")
                    self.assertEqual(label_info['is_training_data'], expected_training_status,
                                   f"Failed for label '{label}': expected training_data={expected_training_status}")
                else:
                    # Unlabeled images should be included by default in training logic
                    # (They won't have a database entry, but the dataset should include them)
                    self.assertIsNone(label, "Only unlabeled images should have no database entry")
        finally:
            # Clean up test files
            for img_path in test_files:
                if os.path.exists(img_path):
                    os.remove(img_path)
            if os.path.exists(test_dir):
                shutil.rmtree(test_dir)
                    
    def test_manual_labeling_affects_training_exclusions(self):
        """Test that manual labeling actually changes what gets excluded from training."""
        # Create isolated test directory
        test_dir = os.path.join(self.temp_dir.name, "test_exclusions")
        os.makedirs(test_dir, exist_ok=True)
        
        # Start with some images
        initial_images = ["good1.jpg", "good2.jpg", "false_pos1.jpg", "false_pos2.jpg"]
        
        # Create test files in our isolated directory
        test_paths = []
        for img_name in initial_images:
            img_path = os.path.join(test_dir, img_name)
            Path(img_path).touch()
            test_paths.append(img_path)
        
        try:
            # Initially, no exclusions (all images are "innocent")
            stats = get_training_data_stats(from_directory=test_dir)
            initial_excluded = stats.get('total_excluded', 0)
            
            # Mark some as false positives
            add_image_label(test_paths[2], "not_a_crow", confidence=0.90)  # false_pos1.jpg
            add_image_label(test_paths[3], "not_a_crow", confidence=0.85)  # false_pos2.jpg
            
            # Confirm some as good crows
            add_image_label(test_paths[0], "crow", confidence=0.95)  # good1.jpg
            add_image_label(test_paths[1], "crow", confidence=0.93)  # good2.jpg
            
            # Check that exclusions increased
            updated_stats = get_training_data_stats(from_directory=test_dir)
            final_excluded = updated_stats.get('total_excluded', 0)
            
            self.assertGreater(final_excluded, initial_excluded, 
                              "Manual labeling should increase exclusions")
            self.assertEqual(final_excluded, 2, 
                            "Should exclude exactly 2 false positives")
        finally:
            # Clean up test files
            for img_path in test_paths:
                if os.path.exists(img_path):
                    os.remove(img_path)
            if os.path.exists(test_dir):
                shutil.rmtree(test_dir)
            
    @patch('old_scripts.train_triplet_resnet.logger')
    def test_training_dataset_logging_includes_manual_stats(self, mock_logger):
        """Test that training initialization logs manual labeling statistics."""
        test_files = ["crow1.jpg", "crow2.jpg", "false_pos.jpg"]
        
        # Create test files
        for img_path in test_files:
            Path(img_path).touch()
        
        try:
            # Add some manual labels
            add_image_label("crow1.jpg", "crow", confidence=0.95)
            add_image_label("crow2.jpg", "crow", confidence=0.90)
            add_image_label("false_pos.jpg", "not_a_crow", confidence=0.85)
            
            try:
                # Try to trigger dataset creation (this will call our logging code)
                from old_scripts.train_triplet_resnet import CrowTripletDataset
                dataset = CrowTripletDataset(self.crop_dir)
                
                # Check that appropriate logging calls were made
                # Look for calls that mention manual labeling
                log_calls = [str(call) for call in mock_logger.info.call_args_list]
                manual_log_calls = [call for call in log_calls if 'manual' in call.lower() or 'labeling' in call.lower()]
                
                # Should have at least one log call about manual labeling
                self.assertGreater(len(manual_log_calls), 0, 
                                 "Should log information about manual labeling statistics")
                                 
            except ImportError:
                self.skipTest("CrowTripletDataset not available for testing")
        finally:
            # Clean up test files
            for img_path in test_files:
                if os.path.exists(img_path):
                    os.remove(img_path)
            
    def test_dataset_size_changes_with_manual_labeling(self):
        """Test that dataset size changes as images are manually excluded."""
        try:
            from old_scripts.train_triplet_resnet import CrowTripletDataset
            
            # Initial dataset size (all images included)
            initial_dataset = CrowTripletDataset(self.crop_dir)
            initial_size = len(initial_dataset)
            
            # Exclude some images
            add_image_label(self.test_images[0], "not_a_crow", confidence=0.90)
            add_image_label(self.test_images[1], "not_a_crow", confidence=0.85)
            
            # Create new dataset (should be smaller)
            filtered_dataset = CrowTripletDataset(self.crop_dir)
            filtered_size = len(filtered_dataset)
            
            # Dataset should be smaller after exclusions
            self.assertLess(filtered_size, initial_size, 
                           "Dataset should shrink when images are excluded")
            # Note: The exact difference might vary due to directory structure
            # Just check that it's smaller
            
        except ImportError:
            self.skipTest("CrowTripletDataset not available for testing")
            
    def test_unlabeled_images_remain_included(self):
        """Test that unlabeled images are still included in training (innocent until proven guilty)."""
        try:
            from old_scripts.train_triplet_resnet import CrowTripletDataset
            
            # Label some images but leave others unlabeled
            add_image_label(self.test_images[0], "crow", confidence=0.95)
            add_image_label(self.test_images[1], "not_a_crow", confidence=0.90)
            # test_images[2], [3], [4] remain unlabeled
            
            # Create dataset
            dataset = CrowTripletDataset(self.crop_dir)
            dataset_paths = [str(sample[0]) for sample in dataset.samples]
            included_basenames = [os.path.basename(path) for path in dataset_paths]
            
            # Unlabeled images should still be included
            self.assertIn("test_image_2.jpg", included_basenames)
            self.assertIn("test_image_3.jpg", included_basenames)  
            self.assertIn("test_image_4.jpg", included_basenames)
            
            # Confirmed crow should be included
            self.assertIn("test_image_0.jpg", included_basenames)
            
            # False positive should be excluded
            self.assertNotIn("test_image_1.jpg", included_basenames)
            
        except ImportError:
            self.skipTest("CrowTripletDataset not available for testing")


class TestImageReviewerIntegration(unittest.TestCase):
    """Test integration between image reviewer and database."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.db_path = os.path.join(cls.temp_dir.name, "test_reviewer_integration.db")
        os.environ['CROW_DB_PATH'] = cls.db_path
        
    def setUp(self):
        """Set up each test."""
        initialize_database()
        
        # Clear any existing image labels from previous tests
        from db import get_connection
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM image_labels")
        conn.commit()
        conn.close()
        
    @classmethod
    def tearDownClass(cls):
        """Clean up test fixtures."""
        cls.temp_dir.cleanup()
        
    def test_image_reviewer_database_persistence(self):
        """Test that image reviewer labels persist in database."""
        # Simulate image reviewer labeling
        test_image = "reviewer_test.jpg"
        
        # Create test file
        Path(test_image).touch()
        
        try:
            # Add label as image reviewer would
            result = add_image_label(test_image, "not_a_crow", confidence=0.85,
                                   reviewer_notes="Reviewed by human - definitely not a crow")
            self.assertTrue(result)
            
            # Verify persistence
            label_info = get_image_label(test_image)
            self.assertIsNotNone(label_info)
            self.assertEqual(label_info['label'], "not_a_crow")
            self.assertEqual(label_info['confidence'], 0.85)
            self.assertIn("human", label_info['reviewer_notes'].lower())
            self.assertFalse(label_info['is_training_data'])  # Should be excluded from training
        finally:
            # Clean up test file
            if os.path.exists(test_image):
                os.remove(test_image)
        
    def test_image_reviewer_updates_existing_labels(self):
        """Test that image reviewer can update existing labels."""
        test_image = "update_test.jpg"
        
        # Create test file
        Path(test_image).touch()
        
        try:
            # Initial label (perhaps from previous review)
            add_image_label(test_image, "not_sure", confidence=0.70,
                           reviewer_notes="Initial uncertain assessment")
            
            # Updated label after more careful review
            add_image_label(test_image, "crow", confidence=0.95,
                           reviewer_notes="Actually a confirmed crow after closer inspection")
            
            # Verify update
            final_info = get_image_label(test_image)
            self.assertEqual(final_info['label'], "crow")
            self.assertEqual(final_info['confidence'], 0.95)
            self.assertIn("confirmed crow", final_info['reviewer_notes'].lower())
            self.assertTrue(final_info['is_training_data'])  # Should be included in training
        finally:
            # Clean up test file
            if os.path.exists(test_image):
                os.remove(test_image)


if __name__ == '__main__':
    unittest.main() 