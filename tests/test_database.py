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
    clear_database
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
        self.assertEqual(len(history), 1)  # One embedding
        
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
        
        # Create similar embedding
        embedding2 = embedding1 + np.random.rand(512).astype(np.float32) * 0.1
        
        # Try to match similar embedding
        crow_id2 = save_crow_embedding(embedding2, "test.mp4", 101, 0.95)
        
        # Should match the same crow
        self.assertEqual(crow_id1, crow_id2)
        
        # Create very different embedding
        embedding3 = np.random.rand(512).astype(np.float32)
        
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
        # Remove key file if it exists
        if os.path.exists(self.key_path):
            os.remove(self.key_path)
            
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
        # Get key (should generate new one)
        key1 = get_encryption_key()
        self.assertIsNotNone(key1)
        self.assertEqual(len(key1), 32)  # Fernet key length
        
        # Verify key file was created
        self.assertTrue(os.path.exists(self.key_path))
        
        # Get key again (should retrieve existing one)
        key2 = get_encryption_key()
        self.assertEqual(key1, key2)
        
    def test_secure_database_connection(self):
        """Test secure database connection."""
        # Create test database
        initialize_database()
        
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
        with self.assertRaises(Exception):
            secure_database_connection("nonexistent.db")

if __name__ == '__main__':
    unittest.main() 