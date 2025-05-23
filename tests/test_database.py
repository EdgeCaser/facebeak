import unittest
from unittest.mock import MagicMock, patch, call
import pytest
import sqlite3
import os
import tempfile
from pathlib import Path
import numpy as np
from db import (
    initialize_database,
    get_connection,
    save_crow_embedding,
    find_matching_crow,
    get_crow_history,
    get_all_crows,
    update_crow_name,
    get_crow_embeddings,
    add_behavioral_marker,
    get_segment_markers,
    backup_database,
    clear_database,
    add_image_label,
    get_image_label,
    get_unlabeled_images,
    get_training_data_stats,
    get_training_suitable_images
)
from db_security import get_encryption_key, secure_database_connection

class TestDatabaseOperations(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures that are shared across all tests."""
        # Create a temporary directory for test database
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.db_path = os.path.join(cls.temp_dir.name, "test_crow_embeddings.db")
        
        # Set environment variable for test database
        os.environ['CROW_DB_PATH'] = cls.db_path
        
    def setUp(self):
        """Set up test fixtures that are run before each test."""
        # Initialize fresh database for each test
        initialize_database()
        
        # Clear any existing markers
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM behavioral_markers")
        conn.commit()
        conn.close()
        
    def tearDown(self):
        """Clean up after each test."""
        # Clear database after each test
        clear_database()
        
    @classmethod
    def tearDownClass(cls):
        """Clean up test fixtures."""
        cls.temp_dir.cleanup()
        
    def test_database_initialization(self):
        """Test database initialization."""
        # Verify tables exist
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cursor.fetchall()}
        
        self.assertIn('crows', tables)
        self.assertIn('crow_embeddings', tables)
        self.assertIn('behavioral_markers', tables)
        
        conn.close()
        
    def test_crow_operations(self):
        """Test crow record operations."""
        # Create test embedding
        embedding = np.random.rand(512).astype(np.float32)
        
        # Save embedding (creates new crow)
        crow_id = save_crow_embedding(embedding, "test.mp4", 100, 0.95)
        self.assertIsNotNone(crow_id)
        
        # Get crow history
        history = get_crow_history(crow_id)
        self.assertIsNotNone(history)
        self.assertEqual(len(history['embeddings']), 1)  # One embedding
        
        # Update crow name
        update_crow_name(crow_id, "test_crow")
        
        # Get all crows
        crows = get_all_crows()
        self.assertEqual(len(crows), 1)
        self.assertEqual(crows[0]['id'], crow_id)
        
        # Get crow embeddings
        embeddings = get_crow_embeddings(crow_id)
        self.assertEqual(len(embeddings), 1)
        np.testing.assert_array_almost_equal(embeddings[0]['embedding'], embedding)
        
    def test_behavioral_markers(self):
        """Test behavioral marker operations."""
        # Create test embedding and crow
        embedding = np.random.rand(512).astype(np.float32)
        crow_id = save_crow_embedding(embedding, "test.mp4", 100, 0.95)
        
        # Get the segment ID
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT id FROM crow_embeddings WHERE crow_id = ?', (crow_id,))
        segment_id = cursor.fetchone()[0]
        conn.close()
        
        # Add behavioral marker
        marker_id = add_behavioral_marker(
            segment_id,
            "test_marker",
            "Test details",
            confidence=0.9,
            frame_number=100
        )
        self.assertIsNotNone(marker_id)
        
        # Get segment markers
        markers = get_segment_markers(segment_id)
        self.assertEqual(len(markers), 1)
        self.assertEqual(markers[0]['marker_type'], "test_marker")
        
    def test_crow_matching(self):
        """Test crow matching functionality."""
        # Create test embedding
        embedding1 = np.random.rand(512).astype(np.float32)
        
        # Save first embedding (creates new crow)
        crow_id1 = save_crow_embedding(embedding1, "test.mp4", 100, 0.95)
        
        # Create similar embedding (but not too similar)
        embedding2 = embedding1 + np.random.rand(512).astype(np.float32) * 0.3  # Increased noise
        
        # Try to match similar embedding
        crow_id2 = save_crow_embedding(embedding2, "test.mp4", 101, 0.95)
        
        # Should match the same crow
        self.assertEqual(crow_id1, crow_id2)
        
        # Create very different embedding (negative of the original)
        embedding3 = -embedding1
        embedding3 = embedding3 / np.linalg.norm(embedding3)  # Normalize
        
        # Try to match different embedding
        crow_id3 = save_crow_embedding(embedding3, "test.mp4", 102, 0.95)
        
        # Should create new crow
        self.assertNotEqual(crow_id1, crow_id3)
        
    def test_backup_database(self):
        """Test database backup functionality."""
        # Add some test data
        embedding = np.random.rand(512).astype(np.float32)
        save_crow_embedding(embedding, "test.mp4", 100, 0.95)
        
        # Create backup
        backup_path = backup_database()
        
        # Verify backup exists
        self.assertTrue(os.path.exists(backup_path))
        
        # Clear original database
        clear_database()
        
        # Verify database is empty
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM crows")
        self.assertEqual(cursor.fetchone()[0], 0)
        conn.close()

    def test_add_image_label(self):
        """Test adding and updating image labels."""
        test_path = "test_images/crow_001.jpg"
        
        # Create test file
        os.makedirs(os.path.dirname(test_path), exist_ok=True)
        Path(test_path).touch()
        
        try:
            # Add initial label
            result = add_image_label(test_path, "crow", confidence=0.95, 
                                   reviewer_notes="Confirmed crow")
            self.assertTrue(result)
            
            # Retrieve label
            label_info = get_image_label(test_path)
            self.assertIsNotNone(label_info)
            self.assertEqual(label_info['label'], "crow")
            self.assertEqual(label_info['confidence'], 0.95)
            self.assertEqual(label_info['reviewer_notes'], "Confirmed crow")
            self.assertTrue(label_info['is_training_data'])
            
            # Update label
            result = add_image_label(test_path, "not_a_crow", confidence=0.90,
                                   reviewer_notes="Actually not a crow")
            self.assertTrue(result)
            
            # Verify update
            updated_info = get_image_label(test_path)
            self.assertEqual(updated_info['label'], "not_a_crow")
            self.assertEqual(updated_info['confidence'], 0.90)
            self.assertFalse(updated_info['is_training_data'])  # Should be excluded from training
        finally:
            # Clean up test file and directory
            if os.path.exists(test_path):
                os.remove(test_path)
            try:
                os.rmdir(os.path.dirname(test_path))
            except OSError:
                pass  # Directory not empty or doesn't exist
        
    def test_get_unlabeled_images(self):
        """Test fetching unlabeled images with various filters."""
        # Create some test image paths (simulating real crop directory structure)
        test_images = [
            "crow_crops/123/test1.jpg",
            "crow_crops/124/test2.jpg", 
            "crow_crops/125/test3.jpg",
            "crow_crops/126/test4.jpg"
        ]
        
        # Create mock files for testing
        for img_path in test_images:
            os.makedirs(os.path.dirname(img_path), exist_ok=True)
            Path(img_path).touch()
        
        try:
            # Label some images
            add_image_label(test_images[0], "crow")
            add_image_label(test_images[1], "not_a_crow")
            # Leave test_images[2] and test_images[3] unlabeled
            
            # Test getting unlabeled images
            unlabeled = get_unlabeled_images(limit=10, from_directory="crow_crops")
            
            # Should only return unlabeled images
            unlabeled_basenames = [os.path.basename(path) for path in unlabeled]
            self.assertIn("test3.jpg", unlabeled_basenames)
            self.assertIn("test4.jpg", unlabeled_basenames)
            self.assertNotIn("test1.jpg", unlabeled_basenames)  # Labeled as crow
            self.assertNotIn("test2.jpg", unlabeled_basenames)  # Labeled as not_a_crow
            
            # Test limit functionality
            limited = get_unlabeled_images(limit=1, from_directory="crow_crops")
            self.assertLessEqual(len(limited), 1)
            
        finally:
            # Clean up test files
            for img_path in test_images:
                if os.path.exists(img_path):
                    os.remove(img_path)
                # Remove directories if empty
                try:
                    os.rmdir(os.path.dirname(img_path))
                except OSError:
                    pass  # Directory not empty, that's ok
            try:
                os.rmdir("crow_crops")
            except OSError:
                pass  # Directory not empty or doesn't exist
                
    def test_training_data_stats(self):
        """Test manual labeling statistics calculation."""
        # Add various labels
        test_images = [
            ("img1.jpg", "crow", 0.95),
            ("img2.jpg", "crow", 0.90), 
            ("img3.jpg", "not_a_crow", 0.85),
            ("img4.jpg", "not_a_crow", 0.88),
            ("img5.jpg", "not_sure", 0.70),
            ("img6.jpg", "crow", 0.92)
        ]
        
        for img_path, label, confidence in test_images:
            add_image_label(img_path, label, confidence=confidence)
        
        # Get statistics
        stats = get_training_data_stats()
        
        # Verify counts
        self.assertEqual(stats['crow']['count'], 3)
        self.assertEqual(stats['not_a_crow']['count'], 2)
        self.assertEqual(stats['not_sure']['count'], 1)
        self.assertEqual(stats['total_labeled'], 6)
        self.assertEqual(stats['total_excluded'], 2)  # Only not_a_crow are excluded
        
        # Verify average confidences
        self.assertAlmostEqual(stats['crow']['avg_confidence'], (0.95 + 0.90 + 0.92) / 3, places=2)
        self.assertAlmostEqual(stats['not_a_crow']['avg_confidence'], (0.85 + 0.88) / 2, places=2)
        
    def test_get_training_suitable_images(self):
        """Test filtering logic for training data based on manual labels."""
        test_images = [
            "good_crow_1.jpg",
            "good_crow_2.jpg", 
            "false_positive.jpg",
            "uncertain.jpg",
            "unlabeled.jpg"
        ]
        
        # Label images with different classifications
        add_image_label(test_images[0], "crow", confidence=0.95)
        add_image_label(test_images[1], "crow", confidence=0.90)
        add_image_label(test_images[2], "not_a_crow", confidence=0.85)  # Should be excluded
        add_image_label(test_images[3], "not_sure", confidence=0.70)   # Should be included
        # test_images[4] remains unlabeled - should be included
        
        # Get training suitable images
        suitable = get_training_suitable_images()
        suitable_basenames = [os.path.basename(path) for path in suitable]
        
        # Verify inclusion/exclusion logic
        self.assertIn("good_crow_1.jpg", suitable_basenames)
        self.assertIn("good_crow_2.jpg", suitable_basenames)
        self.assertNotIn("false_positive.jpg", suitable_basenames)  # Excluded
        self.assertIn("uncertain.jpg", suitable_basenames)  # Included (benefit of doubt)
        # Note: unlabeled.jpg won't appear unless it exists in the filesystem
        
    def test_innocent_until_proven_guilty_philosophy(self):
        """Test that images are included unless explicitly marked as not_a_crow."""
        test_cases = [
            ("unlabeled.jpg", None, True),  # No label = included
            ("confirmed_crow.jpg", "crow", True),  # Labeled crow = included
            ("uncertain_bird.jpg", "not_sure", True),  # Not sure = included (benefit of doubt)
            ("false_positive.jpg", "not_a_crow", False),  # Not a crow = excluded
        ]
        
        # Apply labels (skip unlabeled case)
        for img_path, label, should_include in test_cases:
            if label is not None:
                add_image_label(img_path, label, confidence=0.8)
        
        # Check training suitability for labeled images
        suitable = get_training_suitable_images()
        suitable_basenames = [os.path.basename(path) for path in suitable]
        
        # Test labeled images
        for img_path, label, should_include in test_cases[1:]:  # Skip unlabeled test
            if should_include:
                # Note: Image won't appear in suitable list unless it exists in filesystem
                # But we can test the database logic
                label_info = get_image_label(img_path)
                if label_info:
                    expected_training_status = label != "not_a_crow"
                    self.assertEqual(label_info['is_training_data'], expected_training_status)
                    
    def test_image_label_edge_cases(self):
        """Test edge cases for image labeling."""
        # Test empty path - should raise exception
        with self.assertRaises(ValueError):
            add_image_label("", "crow")
        
        # Test invalid label
        test_file = "test_invalid.jpg"
        Path(test_file).touch()
        try:
            with self.assertRaises(ValueError):
                add_image_label(test_file, "invalid_label")
        finally:
            if os.path.exists(test_file):
                os.remove(test_file)
        
        # Test negative confidence
        test_file = "test_negative.jpg"
        Path(test_file).touch()
        try:
            with self.assertRaises(ValueError):
                add_image_label(test_file, "crow", confidence=-0.1)
        finally:
            if os.path.exists(test_file):
                os.remove(test_file)
        
        # Test confidence > 1
        test_file = "test_high_conf.jpg"
        Path(test_file).touch()
        try:
            with self.assertRaises(ValueError):
                add_image_label(test_file, "crow", confidence=1.5)
        finally:
            if os.path.exists(test_file):
                os.remove(test_file)
        
        # Test very long reviewer notes
        test_file = "test_long_notes.jpg"
        Path(test_file).touch()
        try:
            long_notes = "A" * 2000  # Very long string
            result = add_image_label(test_file, "crow", reviewer_notes=long_notes)
            self.assertTrue(result)  # Should still work
            
            # Verify notes were stored (possibly truncated)
            label_info = get_image_label(test_file)
            self.assertIsNotNone(label_info)
            self.assertIsNotNone(label_info['reviewer_notes'])
        finally:
            if os.path.exists(test_file):
                os.remove(test_file)
        
    def test_database_image_labels_table_structure(self):
        """Test that the image_labels table has the correct structure."""
        conn = get_connection()
        cursor = conn.cursor()
        
        # Check table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='image_labels'")
        self.assertIsNotNone(cursor.fetchone())
        
        # Check table structure
        cursor.execute("PRAGMA table_info(image_labels)")
        columns = {row[1]: row[2] for row in cursor.fetchall()}  # {column_name: type}
        
        expected_columns = {
            'id': 'INTEGER',
            'image_path': 'TEXT',
            'label': 'TEXT', 
            'confidence': 'FLOAT',
            'reviewer_notes': 'TEXT',
            'is_training_data': 'INTEGER',
            'timestamp': 'TIMESTAMP'
        }
        
        for col_name, col_type in expected_columns.items():
            self.assertIn(col_name, columns)
            
        conn.close()

class TestDatabaseSecurity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures that are shared across all tests."""
        # Create a temporary directory for test database and key
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.db_path = os.path.join(cls.temp_dir.name, "test_crow_embeddings.db")
        cls.key_path = os.path.join(cls.temp_dir.name, "crow_embeddings.key")
        
        # Set environment variables
        os.environ['CROW_DB_PATH'] = cls.db_path
        
    def setUp(self):
        """Set up test fixtures that are run before each test."""
        # Set environment variable for test database
        os.environ['CROW_DB_PATH'] = self.db_path
        # Remove key file if it exists
        if os.path.exists(self.key_path):
            os.remove(self.key_path)
        # Initialize database for connection tests
        initialize_database()
            
    def tearDown(self):
        """Clean up after each test."""
        # Remove key file
        if os.path.exists(self.key_path):
            os.remove(self.key_path)
            
    @classmethod
    def tearDownClass(cls):
        """Clean up test fixtures."""
        cls.temp_dir.cleanup()
        
    def test_get_encryption_key(self):
        """Test encryption key generation and retrieval."""
        # Test with FACEBEAK_KEY_DIR set
        os.environ["FACEBEAK_KEY_DIR"] = self.temp_dir.name
        key1 = get_encryption_key(test_mode=True)
        self.assertIsNotNone(key1)
        self.assertEqual(len(key1), 44)  # Fernet key length is 44 bytes
        
        # Verify key file was created in the specified directory
        self.assertTrue(os.path.exists(self.key_path))
        
        # Get key again (should retrieve existing one)
        key2 = get_encryption_key(test_mode=True)
        self.assertEqual(key1, key2)
        
        # Test without FACEBEAK_KEY_DIR (should use default location)
        del os.environ["FACEBEAK_KEY_DIR"]
        default_key_dir = Path.home() / '.facebeak' / 'keys'
        default_key_path = default_key_dir / 'crow_embeddings.key'
        default_salt_path = default_key_dir / 'crow_embeddings.salt'
        
        # Remove any existing key at default location
        if default_key_path.exists():
            default_key_path.unlink()
        if default_salt_path.exists():
            default_salt_path.unlink()
            
        # Get new key (should use default location)
        key3 = get_encryption_key(test_mode=True)
        self.assertIsNotNone(key3)
        self.assertEqual(len(key3), 44)
        
        # Verify key was created in default location
        self.assertTrue(default_key_path.exists())
        self.assertTrue(default_salt_path.exists())
        
        # Clean up default key files
        default_key_path.unlink()
        default_salt_path.unlink()
        try:
            default_key_dir.rmdir()
        except OSError:
            # Directory might not be empty, that's okay
            pass
        
    def test_secure_database_connection(self):
        """Test secure database connection."""
        # Ensure database file exists at the correct path
        if not os.path.exists(self.db_path):
            # Create an empty SQLite database file
            import sqlite3
            conn = sqlite3.connect(self.db_path)
            conn.close()
        
        # Get secure connection
        conn = secure_database_connection(self.db_path)
        self.assertIsNotNone(conn)
        
        # Verify connection settings
        cursor = conn.cursor()
        cursor.execute("PRAGMA foreign_keys")
        self.assertEqual(cursor.fetchone()[0], 1)
        
        cursor.execute("PRAGMA busy_timeout")
        self.assertEqual(cursor.fetchone()[0], 5000)
        
        conn.close()
        
    def test_secure_connection_error_handling(self):
        """Test error handling in secure connection."""
        # Try to connect to non-existent database
        non_existent_db = os.path.join(self.temp_dir.name, "nonexistent.db")
        with self.assertRaises(FileNotFoundError) as cm:
            secure_database_connection(non_existent_db)
        self.assertIn("Database file not found", str(cm.exception))
        
        # Try to connect to database in non-writable directory
        import sys
        non_writable_dir = os.path.join(self.temp_dir.name, "non_writable")
        os.makedirs(non_writable_dir, exist_ok=True)
        non_writable_db = os.path.join(non_writable_dir, "test.db")
        Path(non_writable_db).touch()
        try:
            os.chmod(non_writable_dir, 0o444)  # Read-only
            # On Windows, os.chmod may not work as expected for directories
            if sys.platform.startswith('win'):
                print("[DEBUG] Skipping non-writable directory test on Windows due to permission model.")
            else:
                with self.assertRaises(FileNotFoundError) as cm:
                    secure_database_connection(non_writable_db)
                self.assertIn("Database file not found", str(cm.exception))
        finally:
            # Restore directory permissions and clean up
            try:
                os.chmod(non_writable_dir, 0o777)
            except Exception:
                pass
            try:
                os.remove(non_writable_db)
            except Exception:
                pass
            try:
                os.rmdir(non_writable_dir)
            except Exception:
                pass

if __name__ == '__main__':
    unittest.main() 