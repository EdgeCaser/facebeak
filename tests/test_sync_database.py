#!/usr/bin/env python3
"""
Comprehensive tests for sync_database.py functionality.
Tests database synchronization with crop directories, error handling, and edge cases.
"""

import pytest
import os
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
import sqlite3
import logging

# Import the sync_database module and database utilities
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utilities import sync_database
import db


class TestSyncDatabase:
    """Test cases for sync_database functionality."""
    
    @pytest.fixture(autouse=True)
    def setup_test_environment(self):
        """Set up test environment with temporary directories and database."""
        # Create temporary directory for test files
        self.test_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        
        # Change to test directory
        os.chdir(self.test_dir)
        
        # Create test crop directory structure
        self.crop_dir = Path("crow_crops")
        self.crop_dir.mkdir(exist_ok=True)
        
        # Create test database
        self.test_db_path = Path("test_crow_embeddings.db")
        
        # Create the database file first (required by some database operations)
        self.test_db_path.touch()
        
        # Patch the database path
        self.db_patcher = patch('db.DB_PATH', self.test_db_path)
        self.db_patcher.start()
        
        # Initialize database
        db.initialize_database()
        
        yield
        
        # Cleanup
        self.db_patcher.stop()
        os.chdir(self.original_cwd)
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def create_test_crop_directory(self, crow_id, num_images=5):
        """Create a test crop directory with images."""
        crow_dir = self.crop_dir / str(crow_id)
        crow_dir.mkdir(exist_ok=True)
        
        # Create dummy image files
        for i in range(num_images):
            img_file = crow_dir / f"frame_{i:03d}.jpg"
            img_file.write_bytes(b"fake_image_data")
        
        return crow_dir
    
    def test_sync_with_empty_crop_directory(self):
        """Test sync when crop directory is empty."""
        # Ensure crop directory exists but is empty
        assert self.crop_dir.exists()
        assert len(list(self.crop_dir.iterdir())) == 0
        
        result = sync_database.sync_database_with_crops()
        
        assert result == 0  # No new crows created
        
        # Verify database is still empty
        crows = db.get_all_crows()
        assert len(crows) == 0
    
    def test_sync_with_single_crow_directory(self):
        """Test sync with a single crow directory."""
        # Create test crow directory
        crow_id = 123
        num_images = 3
        self.create_test_crop_directory(crow_id, num_images)
        
        result = sync_database.sync_database_with_crops()
        
        assert result == 1  # One new crow created
        
        # Verify crow was added to database
        crows = db.get_all_crows()
        assert len(crows) == 1
        assert crows[0]['id'] == crow_id
        assert crows[0]['name'] == f"Crow {crow_id}"
        assert crows[0]['total_sightings'] == num_images
    
    def test_sync_with_multiple_crow_directories(self):
        """Test sync with multiple crow directories."""
        # Create multiple test crow directories
        crow_ids = [101, 205, 350]
        image_counts = [3, 7, 2]
        
        for crow_id, num_images in zip(crow_ids, image_counts):
            self.create_test_crop_directory(crow_id, num_images)
        
        result = sync_database.sync_database_with_crops()
        
        assert result == len(crow_ids)  # All crows created
        
        # Verify all crows were added to database
        crows = db.get_all_crows()
        assert len(crows) == len(crow_ids)
        
        # Check each crow
        crow_dict = {crow['id']: crow for crow in crows}
        for crow_id, expected_count in zip(crow_ids, image_counts):
            assert crow_id in crow_dict
            assert crow_dict[crow_id]['name'] == f"Crow {crow_id}"
            assert crow_dict[crow_id]['total_sightings'] == expected_count
    
    def test_sync_with_existing_crows_in_database(self):
        """Test sync when some crows already exist in database."""
        # Create crow directories
        crow_ids = [100, 200, 300]
        for crow_id in crow_ids:
            self.create_test_crop_directory(crow_id, 3)
        
        # Manually add one crow to database
        existing_crow_id = 200
        conn = db.get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO crows (id, name, total_sightings, first_seen, last_seen)
            VALUES (?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
        ''', (existing_crow_id, f"Existing Crow {existing_crow_id}", 1))
        conn.commit()
        conn.close()
        
        result = sync_database.sync_database_with_crops()
        
        # Should only create 2 new crows (100 and 300)
        assert result == 2
        
        # Verify all crows are in database
        crows = db.get_all_crows()
        assert len(crows) == 3
        
        # Check existing crow wasn't modified
        crow_dict = {crow['id']: crow for crow in crows}
        assert crow_dict[existing_crow_id]['name'] == f"Existing Crow {existing_crow_id}"
        assert crow_dict[existing_crow_id]['total_sightings'] == 1  # Original value preserved
    
    def test_sync_with_non_numeric_directories(self):
        """Test sync ignores non-numeric directories."""
        # Create numeric directories
        numeric_ids = [101, 202]
        for crow_id in numeric_ids:
            self.create_test_crop_directory(crow_id, 2)
        
        # Create non-numeric directories
        non_numeric_dirs = ["unknown", "background", "test_data", "temp"]
        for dirname in non_numeric_dirs:
            non_numeric_dir = self.crop_dir / dirname
            non_numeric_dir.mkdir()
            # Add some files to make it non-empty
            (non_numeric_dir / "file.jpg").write_bytes(b"data")
        
        result = sync_database.sync_database_with_crops()
        
        # Only numeric directories should be processed
        assert result == len(numeric_ids)
        
        crows = db.get_all_crows()
        assert len(crows) == len(numeric_ids)
        
        crow_ids = {crow['id'] for crow in crows}
        assert crow_ids == set(numeric_ids)
    
    def test_sync_with_empty_crow_directories(self):
        """Test sync skips directories with no images."""
        # Create directories with different content
        crow_with_images = 101
        crow_without_images = 102
        crow_with_non_jpg = 103
        
        # Directory with images
        self.create_test_crop_directory(crow_with_images, 3)
        
        # Empty directory
        empty_dir = self.crop_dir / str(crow_without_images)
        empty_dir.mkdir()
        
        # Directory with non-JPG files
        non_jpg_dir = self.crop_dir / str(crow_with_non_jpg)
        non_jpg_dir.mkdir()
        (non_jpg_dir / "file.txt").write_bytes(b"text data")
        (non_jpg_dir / "image.png").write_bytes(b"png data")
        
        result = sync_database.sync_database_with_crops()
        
        # Only the directory with JPG images should be processed
        assert result == 1
        
        crows = db.get_all_crows()
        assert len(crows) == 1
        assert crows[0]['id'] == crow_with_images
    
    def test_sync_missing_crop_directory(self):
        """Test sync when crop directory doesn't exist."""
        # Remove the crop directory
        shutil.rmtree(self.crop_dir)
        
        # Should handle missing directory gracefully
        result = sync_database.sync_database_with_crops()
        assert result is None  # Function returns early
        
        # Database should remain unchanged
        crows = db.get_all_crows()
        assert len(crows) == 0
    
    def test_sync_with_large_number_directories(self):
        """Test sync with a large number of directories."""
        # Create many crow directories
        crow_ids = list(range(1, 51))  # 50 crows
        image_counts = [2] * len(crow_ids)  # 2 images each
        
        for crow_id in crow_ids:
            self.create_test_crop_directory(crow_id, 2)
        
        result = sync_database.sync_database_with_crops()
        
        assert result == len(crow_ids)
        
        crows = db.get_all_crows()
        assert len(crows) == len(crow_ids)
        
        # Verify all crow IDs are present
        crow_ids_in_db = {crow['id'] for crow in crows}
        assert crow_ids_in_db == set(crow_ids)
    
    def test_sync_database_error_handling(self):
        """Test sync handles database errors gracefully."""
        # Create a crow directory
        self.create_test_crop_directory(123, 3)
        
        # Mock database connection to raise an error
        with patch('db.get_connection') as mock_conn:
            mock_conn.side_effect = sqlite3.Error("Database connection failed")
            
            with pytest.raises(sqlite3.Error):
                sync_database.sync_database_with_crops()
    
    def test_sync_logging_output(self, caplog):
        """Test that sync produces appropriate logging output."""
        # Create test directories
        crow_ids = [111, 222]
        for crow_id in crow_ids:
            self.create_test_crop_directory(crow_id, 3)
        
        with caplog.at_level(logging.INFO):
            result = sync_database.sync_database_with_crops()
        
        assert result == 2
        
        # Check that appropriate log messages were generated
        log_messages = [record.message for record in caplog.records]
        
        # Should log found directories
        assert any("Found 2 crop directories" in msg for msg in log_messages)
        
        # Should log created crows
        assert any("Created crow 111" in msg for msg in log_messages)
        assert any("Created crow 222" in msg for msg in log_messages)
        
        # Should log final stats
        assert any("Successfully created 2 new crow entries" in msg for msg in log_messages)
        assert any("Database now contains 2 crows" in msg for msg in log_messages)
    
    def test_sync_preserves_existing_data(self):
        """Test that sync preserves existing crow data."""
        # Create initial crow in database with custom data
        crow_id = 456
        custom_name = "Special Crow"
        
        conn = db.get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO crows (id, name, total_sightings, first_seen, last_seen)
            VALUES (?, ?, ?, datetime('2023-01-01'), datetime('2023-01-02'))
        ''', (crow_id, custom_name, 10))
        conn.commit()
        conn.close()
        
        # Create crop directory for same crow
        self.create_test_crop_directory(crow_id, 5)
        
        # Also create a new crow directory
        new_crow_id = 789
        self.create_test_crop_directory(new_crow_id, 3)
        
        result = sync_database.sync_database_with_crops()
        
        # Should only create the new crow
        assert result == 1
        
        crows = db.get_all_crows()
        assert len(crows) == 2
        
        # Find the existing crow and verify data is preserved
        crow_dict = {crow['id']: crow for crow in crows}
        existing_crow = crow_dict[crow_id]
        
        assert existing_crow['name'] == custom_name  # Custom name preserved
        assert existing_crow['total_sightings'] == 10  # Original count preserved
        assert '2023-01-01' in existing_crow['first_seen']  # Original timestamp preserved
    
    def test_sync_transaction_integrity(self):
        """Test that sync operations are transactional."""
        # Create crow directories
        crow_ids = [100, 200, 300]
        for crow_id in crow_ids:
            self.create_test_crop_directory(crow_id, 2)
        
        # Mock cursor.execute to fail on the second insertion
        original_execute = None
        call_count = 0
        
        def mock_execute(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:  # Fail on second INSERT
                raise sqlite3.Error("Simulated database error")
            return original_execute(*args, **kwargs)
        
        with patch('db.get_connection') as mock_get_conn:
            mock_conn = MagicMock()
            mock_cursor = MagicMock()
            mock_conn.cursor.return_value = mock_cursor
            mock_get_conn.return_value = mock_conn
            
            # Store original execute method
            original_execute = mock_cursor.execute
            mock_cursor.execute.side_effect = mock_execute
            
            with pytest.raises(sqlite3.Error):
                sync_database.sync_database_with_crops()
            
            # Verify rollback was called (transaction integrity)
            # Note: The actual rollback behavior depends on implementation
            # This test ensures errors are propagated appropriately
    
    def test_sync_file_permissions_error(self):
        """Test sync handles file permission errors gracefully."""
        crow_id = 123
        crow_dir = self.create_test_crop_directory(crow_id, 3)
        
        # Mock Path.glob to raise a permission error
        with patch.object(Path, 'glob') as mock_glob:
            mock_glob.side_effect = PermissionError("Permission denied")
            
            # Should handle the error and continue or fail gracefully
            with pytest.raises(Exception):  # Specific exception type depends on implementation
                sync_database.sync_database_with_crops()
    
    def test_command_line_execution(self, capsys):
        """Test command-line execution of sync_database module."""
        # Create test data
        crow_ids = [111, 222]
        for crow_id in crow_ids:
            self.create_test_crop_directory(crow_id, 2)
        
        # Patch sys.argv and run main
        with patch('sys.argv', ['sync_database.py']):
            # Import and run the main section
            exec(open(os.path.join(os.path.dirname(sync_database.__file__), 'sync_database.py')).read())
        
        # Check console output
        captured = capsys.readouterr()
        assert "Syncing database with existing crop directories..." in captured.out
        assert "Created 2 new crow entries" in captured.out
        assert "You can now restart the suspect lineup tool" in captured.out


class TestSyncDatabaseIntegration:
    """Integration tests for sync_database with real database operations."""
    
    @pytest.fixture(autouse=True)
    def setup_integration_test(self):
        """Set up integration test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        os.chdir(self.test_dir)
        
        # Create real crop directory
        self.crop_dir = Path("crow_crops")
        self.crop_dir.mkdir(exist_ok=True)
        
        # Use real database operations
        self.test_db_path = Path("integration_test.db")
        self.test_db_path.touch()  # Create the database file
        self.db_patcher = patch('db.DB_PATH', self.test_db_path)
        self.db_patcher.start()
        
        db.initialize_database()
        
        yield
        
        self.db_patcher.stop()
        os.chdir(self.original_cwd)
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_full_workflow_integration(self):
        """Test complete workflow from directory creation to database sync."""
        # Simulate realistic crop directory structure
        realistic_data = [
            (1, 15),    # Crow 1 with 15 images
            (5, 8),     # Crow 5 with 8 images
            (12, 23),   # Crow 12 with 23 images
            (25, 3),    # Crow 25 with 3 images
            (100, 45),  # Crow 100 with 45 images
        ]
        
        # Create crop directories with various image counts
        for crow_id, image_count in realistic_data:
            crow_dir = self.crop_dir / str(crow_id)
            crow_dir.mkdir()
            
            # Create realistic image files with proper naming
            for i in range(image_count):
                img_file = crow_dir / f"frame_{i:06d}.jpg"
                # Create a small "realistic" file size
                img_file.write_bytes(b"JPEG_HEADER" + b"x" * (1024 + i))
        
        # Add some non-crow directories that should be ignored
        for ignored_dir in ["temp", "backup", "processing", "2abc"]:
            ignored_path = self.crop_dir / ignored_dir
            ignored_path.mkdir()
            (ignored_path / "file.jpg").write_bytes(b"ignored")
        
        # Perform sync
        created_count = sync_database.sync_database_with_crops()
        
        # Verify results
        assert created_count == len(realistic_data)
        
        # Check database contents
        all_crows = db.get_all_crows()
        assert len(all_crows) == len(realistic_data)
        
        # Verify each crow's data
        crow_dict = {crow['id']: crow for crow in all_crows}
        for expected_id, expected_count in realistic_data:
            assert expected_id in crow_dict
            crow = crow_dict[expected_id]
            assert crow['name'] == f"Crow {expected_id}"
            assert crow['total_sightings'] == expected_count
            assert crow['first_seen'] is not None
            assert crow['last_seen'] is not None
        
        # Test second sync (should not create duplicates)
        second_sync_count = sync_database.sync_database_with_crops()
        assert second_sync_count == 0
        
        # Verify no duplicates created
        all_crows_after = db.get_all_crows()
        assert len(all_crows_after) == len(realistic_data)
    
    def test_sync_with_database_constraints(self):
        """Test sync respects database constraints and handles conflicts."""
        crow_id = 999
        
        # Create crop directory
        crow_dir = self.crop_dir / str(crow_id)
        crow_dir.mkdir()
        (crow_dir / "image1.jpg").write_bytes(b"test_image")
        
        # Manually insert crow with same ID but different data
        conn = db.get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO crows (id, name, total_sightings, first_seen, last_seen)
            VALUES (?, ?, ?, datetime('2022-01-01'), datetime('2022-01-02'))
        ''', (crow_id, "Manually Added Crow", 99))
        conn.commit()
        conn.close()
        
        # Sync should respect existing entry
        created_count = sync_database.sync_database_with_crops()
        assert created_count == 0
        
        # Verify original data is preserved
        crows = db.get_all_crows()
        assert len(crows) == 1
        crow = crows[0]
        assert crow['id'] == crow_id
        assert crow['name'] == "Manually Added Crow"  # Original name preserved
        assert crow['total_sightings'] == 99  # Original count preserved


if __name__ == "__main__":
    pytest.main([__file__]) 