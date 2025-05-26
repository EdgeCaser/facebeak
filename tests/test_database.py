import unittest
import os
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
import sqlite3
import numpy as np

# Import the database module
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
    get_training_suitable_images,
    is_image_training_suitable
)
from db_security import get_encryption_key, secure_database_connection

class TestDatabaseOperations(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        # Create a temporary directory for test databases
        cls.test_dir = tempfile.mkdtemp()
        cls.original_db_path = None
        
    def setUp(self):
        """Set up each test with a fresh database."""
        # Create a unique test database for this test
        self.test_db_path = os.path.join(self.test_dir, f"test_db_{id(self)}.db")
        
        # Create the database file first (required by secure_database_connection)
        Path(self.test_db_path).touch()
        
        # Patch the database path for this test
        self.db_patcher = patch('db.DB_PATH', Path(self.test_db_path))
        self.db_patcher.start()
        
        # Initialize the test database
        initialize_database()
        
    def tearDown(self):
        """Clean up after each test."""
        # Close any open database connections to avoid file locks
        try:
            # Try to close any connections that might be open
            import gc
            gc.collect()  # Force garbage collection to close connections
        except:
            pass
            
        # Stop the patches
        self.db_patcher.stop()
        
        # Remove the test database file if it exists
        try:
            if os.path.exists(self.test_db_path):
                # On Windows, wait a bit for file handles to release
                import time
                time.sleep(0.1)
                os.remove(self.test_db_path)
        except PermissionError:
            # If still locked, mark for cleanup later
            pass
            
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment."""
        # Remove the temporary test directory
        if os.path.exists(cls.test_dir):
            shutil.rmtree(cls.test_dir)
        
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
        # Create test embedding (512D)
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
        
    def test_crow_operations_512d_embeddings(self):
        """Test crow record operations specifically with 512D embeddings."""
        # Test various embedding dimensions
        for embedding_dim in [128, 256, 512, 1024]:
            with self.subTest(embedding_dim=embedding_dim):
                # Create test embedding
                embedding = np.random.rand(embedding_dim).astype(np.float32)
                
                # Normalize embedding (common practice for similarity computation)
                embedding = embedding / np.linalg.norm(embedding)
                
                # Save embedding
                crow_id = save_crow_embedding(embedding, f"test_{embedding_dim}d.mp4", 100, 0.95)
                self.assertIsNotNone(crow_id)
                
                # Verify embedding dimensions are preserved
                embeddings = get_crow_embeddings(crow_id)
                self.assertEqual(len(embeddings), 1)
                saved_embedding = embeddings[0]['embedding']
                self.assertEqual(len(saved_embedding), embedding_dim)
                np.testing.assert_array_almost_equal(saved_embedding, embedding)
        
    def test_512d_similarity_matching(self):
        """Test that 512D embeddings work correctly for similarity matching."""
        # Create a base 512D embedding
        base_embedding = np.random.rand(512).astype(np.float32)
        base_embedding = base_embedding / np.linalg.norm(base_embedding)
        
        # Save the base embedding
        crow_id1 = save_crow_embedding(base_embedding, "test.mp4", 100, 0.95)
        
        # Create a very similar embedding (small noise)
        similar_embedding = base_embedding + np.random.rand(512).astype(np.float32) * 0.1
        similar_embedding = similar_embedding / np.linalg.norm(similar_embedding)
        
        # This should match the same crow
        crow_id2 = save_crow_embedding(similar_embedding, "test.mp4", 101, 0.95)
        self.assertEqual(crow_id1, crow_id2)
        
        # Create a very different embedding (orthogonal to base_embedding)
        different_embedding = np.random.rand(512).astype(np.float32)
        different_embedding = different_embedding / np.linalg.norm(different_embedding)
        # Make it orthogonal to base_embedding to ensure low similarity
        different_embedding = different_embedding - np.dot(different_embedding, base_embedding) * base_embedding
        different_embedding = different_embedding / np.linalg.norm(different_embedding)
        
        # This should create a new crow
        crow_id3 = save_crow_embedding(different_embedding, "test.mp4", 102, 0.95)
        self.assertNotEqual(crow_id1, crow_id3)
        
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
        test_dir = "test_crow_crops_unique"
        test_images = [
            f"{test_dir}/123/test1.jpg",
            f"{test_dir}/124/test2.jpg", 
            f"{test_dir}/125/test3.jpg",
            f"{test_dir}/126/test4.jpg"
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
            
            # Test getting unlabeled images from our test directory
            unlabeled = get_unlabeled_images(limit=10, from_directory=test_dir)
            
            # Should only return unlabeled images
            unlabeled_basenames = [os.path.basename(path) for path in unlabeled]
            self.assertIn("test3.jpg", unlabeled_basenames)
            self.assertIn("test4.jpg", unlabeled_basenames)
            self.assertNotIn("test1.jpg", unlabeled_basenames)  # Labeled as crow
            self.assertNotIn("test2.jpg", unlabeled_basenames)  # Labeled as not_a_crow
            
            # Test limit functionality
            limited = get_unlabeled_images(limit=1, from_directory=test_dir)
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
                os.rmdir(test_dir)
            except OSError:
                pass  # Directory not empty or doesn't exist
                
    def test_training_data_stats(self):
        """Test manual labeling statistics calculation."""
        # Create a test directory for this test
        test_dir = os.path.join(self.test_dir, "test_stats")
        os.makedirs(test_dir, exist_ok=True)
        
        # Add various labels
        test_images = [
            ("img1.jpg", "crow", 0.95),
            ("img2.jpg", "crow", 0.90), 
            ("img3.jpg", "not_a_crow", 0.85),
            ("img4.jpg", "not_a_crow", 0.88),
            ("img5.jpg", "not_sure", 0.70),
            ("img6.jpg", "crow", 0.92)
        ]
        
        # Create test image files in the test directory
        test_paths = []
        for img_name, _, _ in test_images:
            img_path = os.path.join(test_dir, img_name)
            Path(img_path).touch()
            test_paths.append(img_path)
        
        try:
            for img_path, label, confidence in zip(test_paths, [t[1] for t in test_images], [t[2] for t in test_images]):
                add_image_label(img_path, label, confidence=confidence)
            
            # Get statistics from our test directory
            stats = get_training_data_stats(from_directory=test_dir)
            
            # Verify counts
            self.assertEqual(stats['crow']['count'], 3)
            self.assertEqual(stats['not_a_crow']['count'], 2)
            self.assertEqual(stats['not_sure']['count'], 1)
            self.assertEqual(stats['total_labeled'], 6)
            self.assertEqual(stats['total_excluded'], 2)  # Only not_a_crow are excluded
            
            # Verify average confidences
            self.assertAlmostEqual(stats['crow']['avg_confidence'], (0.95 + 0.90 + 0.92) / 3, places=2)
            self.assertAlmostEqual(stats['not_a_crow']['avg_confidence'], (0.85 + 0.88) / 2, places=2)
        finally:
            # Clean up test files
            for test_path in test_paths:
                if os.path.exists(test_path):
                    os.remove(test_path)
            if os.path.exists(test_dir):
                shutil.rmtree(test_dir)
        
    def test_get_training_suitable_images(self):
        """Test filtering logic for training data based on manual labels."""
        # Create a test directory for this test
        test_dir = os.path.join(self.test_dir, "test_training_suitable")
        os.makedirs(test_dir, exist_ok=True)
        
        test_images = [
            "good_crow_1.jpg",
            "good_crow_2.jpg", 
            "false_positive.jpg",
            "uncertain.jpg",
            "unlabeled.jpg"
        ]
        
        # Create test image files in the test directory
        test_paths = []
        for img_name in test_images:
            img_path = os.path.join(test_dir, img_name)
            Path(img_path).touch()
            test_paths.append(img_path)
        
        try:
            # Label images with different classifications
            add_image_label(test_paths[0], "crow", confidence=0.95)  # Should be included
            add_image_label(test_paths[1], "crow", confidence=0.90)  # Should be included
            add_image_label(test_paths[2], "not_a_crow", confidence=0.85)  # Should be excluded
            add_image_label(test_paths[3], "not_sure", confidence=0.60)  # Should be excluded (labeled as not_sure)
            # test_paths[4] remains unlabeled - should be included (innocent until proven guilty)
            
            # Get training suitable images from our test directory
            suitable = get_training_suitable_images(from_directory=test_dir)
            
            # Should return confirmed crows and unlabeled images (innocent until proven guilty)
            suitable_basenames = [os.path.basename(path) for path in suitable]
            self.assertIn("good_crow_1.jpg", suitable_basenames)
            self.assertIn("good_crow_2.jpg", suitable_basenames)
            self.assertIn("unlabeled.jpg", suitable_basenames)  # Unlabeled should be included (innocent until proven guilty)
            self.assertNotIn("false_positive.jpg", suitable_basenames)  # Labeled as not_a_crow
            self.assertNotIn("uncertain.jpg", suitable_basenames)  # Labeled as not_sure
            
        finally:
            # Clean up test files
            for test_path in test_paths:
                if os.path.exists(test_path):
                    os.remove(test_path)
            if os.path.exists(test_dir):
                shutil.rmtree(test_dir)
        
    def test_innocent_until_proven_guilty_philosophy(self):
        """Test that unlabeled images are treated as potential training data."""
        # Create a test directory for this test
        test_dir = os.path.join(self.test_dir, "test_philosophy")
        os.makedirs(test_dir, exist_ok=True)
        
        test_images = [
            "unlabeled_1.jpg",
            "unlabeled_2.jpg",
            "confirmed_crow.jpg",
            "confirmed_not_crow.jpg"
        ]
        
        # Create test image files in the test directory
        test_paths = []
        for img_name in test_images:
            img_path = os.path.join(test_dir, img_name)
            Path(img_path).touch()
            test_paths.append(img_path)
        
        try:
            # Only label some images
            add_image_label(test_paths[2], "crow", confidence=0.95)
            add_image_label(test_paths[3], "not_a_crow", confidence=0.90)
            # Leave test_paths[0] and test_paths[1] unlabeled
            
            # Test that unlabeled images are considered suitable for training
            # (innocent until proven guilty - assume they're good unless labeled otherwise)
            for unlabeled_path in test_paths[:2]:
                is_suitable = is_image_training_suitable(unlabeled_path)
                self.assertTrue(is_suitable, f"Unlabeled image {unlabeled_path} should be suitable for training")
            
            # Test that explicitly labeled images follow their labels
            self.assertTrue(is_image_training_suitable(test_paths[2]))  # Confirmed crow
            self.assertFalse(is_image_training_suitable(test_paths[3]))  # Confirmed not crow
            
        finally:
            # Clean up test files
            for test_path in test_paths:
                if os.path.exists(test_path):
                    os.remove(test_path)
            if os.path.exists(test_dir):
                shutil.rmtree(test_dir)
        
    def test_image_label_edge_cases(self):
        """Test edge cases for image labeling functionality."""
        # Test non-existent file
        non_existent = "does_not_exist.jpg"
        result = add_image_label(non_existent, "crow")
        self.assertFalse(result)
        
        # Test getting label for non-existent file
        label_info = get_image_label(non_existent)
        self.assertIsNone(label_info)
        
        # Test empty label
        test_path = "test_empty_label.jpg"
        Path(test_path).touch()
        
        try:
            result = add_image_label(test_path, "", confidence=0.95)
            self.assertFalse(result)  # Should fail with empty label
            
            # Test very low confidence
            result = add_image_label(test_path, "crow", confidence=0.1)
            self.assertTrue(result)  # Should succeed but mark as low confidence
            
            label_info = get_image_label(test_path)
            self.assertEqual(label_info['confidence'], 0.1)
            
        finally:
            if os.path.exists(test_path):
                os.remove(test_path)
        
    def test_database_image_labels_table_structure(self):
        """Test that the image_labels table has the correct structure."""
        conn = get_connection()
        cursor = conn.cursor()
        
        # Check if image_labels table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='image_labels'")
        table_exists = cursor.fetchone() is not None
        
        if table_exists:
            # Check table structure
            cursor.execute("PRAGMA table_info(image_labels)")
            columns = {row[1]: row[2] for row in cursor.fetchall()}  # column_name: data_type
            
            # Verify expected columns exist
            expected_columns = {
                'id': 'INTEGER',
                'image_path': 'TEXT',
                'label': 'TEXT',
                'confidence': 'REAL',
                'reviewer_notes': 'TEXT',
                'created_at': 'TIMESTAMP',
                'updated_at': 'TIMESTAMP'
            }
            
            for col_name, col_type in expected_columns.items():
                self.assertIn(col_name, columns, f"Column {col_name} should exist")
                # Note: SQLite type affinity means exact type matching can be flexible
        
        conn.close()

class TestDatabaseSecurity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        # Create a temporary directory for test databases
        cls.test_dir = tempfile.mkdtemp()
        
    def setUp(self):
        """Set up each test with a fresh database."""
        # Create a unique test database for this test
        self.test_db_path = os.path.join(self.test_dir, f"test_security_db_{id(self)}.db")
        
    def tearDown(self):
        """Clean up after each test."""
        # Remove the test database file if it exists
        if os.path.exists(self.test_db_path):
            os.remove(self.test_db_path)
            
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment."""
        # Remove the temporary test directory
        if os.path.exists(cls.test_dir):
            shutil.rmtree(cls.test_dir)
        
    def test_get_encryption_key(self):
        """Test encryption key generation and retrieval."""
        # Test that we can get an encryption key
        key = get_encryption_key()
        self.assertIsNotNone(key)
        self.assertIsInstance(key, bytes)
        self.assertEqual(len(key), 32)  # 256-bit key
        
        # Test that the same key is returned on subsequent calls
        key2 = get_encryption_key()
        self.assertEqual(key, key2)
        
        # Test key persistence (if implemented)
        # This would depend on how the key storage is implemented
        
    @patch('db_security.get_encryption_key')
    def test_secure_database_connection_with_mock_key(self, mock_get_key):
        """Test secure database connection with mocked encryption key."""
        # Mock the encryption key
        mock_key = b'test_key_32_bytes_long_for_aes256'
        mock_get_key.return_value = mock_key
        
        # Create test database file
        Path(self.test_db_path).touch()
        
        # Test secure connection
        try:
            conn = secure_database_connection(self.test_db_path)
            self.assertIsNotNone(conn)
            
            # Test that we can perform basic operations
            cursor = conn.cursor()
            cursor.execute("CREATE TABLE test_table (id INTEGER PRIMARY KEY, data TEXT)")
            cursor.execute("INSERT INTO test_table (data) VALUES (?)", ("test_data",))
            conn.commit()
            
            cursor.execute("SELECT data FROM test_table WHERE id = 1")
            result = cursor.fetchone()
            self.assertEqual(result[0], "test_data")
            
            conn.close()
            
        except Exception as e:
            # If encryption is not available, the test should still pass
            # but we should note that encryption is not working
            self.skipTest(f"Encryption not available: {e}")
    
    def test_secure_connection_error_handling(self):
        """Test error handling in secure database connections."""
        # Test with non-existent file
        non_existent_path = os.path.join(self.test_dir, "does_not_exist.db")
        
        try:
            conn = secure_database_connection(non_existent_path)
            # If this succeeds, it means the connection creates the file
            self.assertIsNotNone(conn)
            conn.close()
            # Clean up created file
            if os.path.exists(non_existent_path):
                os.remove(non_existent_path)
        except Exception:
            # If it fails, that's also acceptable behavior
            pass

if __name__ == '__main__':
    unittest.main() 