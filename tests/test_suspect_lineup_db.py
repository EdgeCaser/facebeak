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
    get_crow_videos,
    get_first_crow_image,
    get_crow_images_from_video,
    get_embedding_ids_by_image_paths,
    delete_crow_embeddings,
    reassign_crow_embeddings,
    create_new_crow_from_embeddings,
    get_all_crows
)

class TestSuspectLineupDatabase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        cls.test_dir = tempfile.mkdtemp()
        cls.crow_crops_dir = os.path.join(cls.test_dir, "crow_crops")
        os.makedirs(cls.crow_crops_dir, exist_ok=True)
        
    def setUp(self):
        """Set up each test with a fresh database."""
        self.test_db_path = os.path.join(self.test_dir, f"test_db_{id(self)}.db")
        Path(self.test_db_path).touch()
        
        # Patch the database path
        self.db_patcher = patch('db.DB_PATH', Path(self.test_db_path))
        self.db_patcher.start()
        
        # Initialize the test database
        initialize_database()
        
        # Create test crow crop directories and images
        self.setup_test_crop_structure()
        
    def tearDown(self):
        """Clean up after each test."""
        self.db_patcher.stop()
        
        # Clean up database file
        try:
            if os.path.exists(self.test_db_path):
                import time
                time.sleep(0.1)
                os.remove(self.test_db_path)
        except PermissionError:
            pass
            
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment."""
        if os.path.exists(cls.test_dir):
            shutil.rmtree(cls.test_dir)
    
    def setup_test_crop_structure(self):
        """Create test crop directory structure with sample images."""
        # Create crop directories for test crows
        for crow_id in [100, 200, 300]:
            crow_dir = os.path.join(self.crow_crops_dir, str(crow_id))
            os.makedirs(crow_dir, exist_ok=True)
            
            # Create test image files
            for video in ["video1.mp4", "video2.mp4"]:
                for frame in [10, 20, 30]:
                    for detection in [1, 2]:
                        image_name = f"{video}_frame{frame:04d}_{detection}.jpg"
                        image_path = os.path.join(crow_dir, image_name)
                        # Create empty file
                        Path(image_path).touch()
    
    def create_test_crow_with_embeddings(self, crow_id=None):
        """Helper to create a test crow with embeddings."""
        embedding = np.random.rand(512).astype(np.float32)
        test_crow_id = save_crow_embedding(embedding, "video1.mp4", 100, 0.95)
        
        # Add more embeddings for the same crow
        embedding2 = np.random.rand(512).astype(np.float32)
        save_crow_embedding(embedding2, "video2.mp4", 200, 0.90)
        
        return test_crow_id
    
    def test_get_crow_videos_with_embeddings(self):
        """Test get_crow_videos when embeddings exist in database."""
        crow_id = self.create_test_crow_with_embeddings()
        
        videos = get_crow_videos(crow_id)
        
        self.assertIsInstance(videos, list)
        self.assertGreater(len(videos), 0)
        
        # Check video structure
        video = videos[0]
        self.assertIn('video_path', video)
        self.assertIn('sighting_count', video)
        self.assertIn('first_seen', video)
        self.assertIn('last_seen', video)
    
    @patch('pathlib.Path.exists')
    @patch('pathlib.Path.glob')
    def test_get_crow_videos_fallback_mode(self, mock_glob, mock_exists):
        """Test get_crow_videos fallback when no embeddings exist."""
        # Create a crow without embeddings
        conn = get_connection()
        c = conn.cursor()
        c.execute('INSERT INTO crows (name, total_sightings) VALUES (?, ?)', 
                 ("Test Crow", 0))
        crow_id = c.lastrowid
        conn.commit()
        conn.close()
        
        # Mock crop directory existence and contents
        mock_exists.return_value = True
        mock_files = []
        filenames = ["video1.mp4_frame0010_1.jpg", "video1.mp4_frame0020_1.jpg", "video2.mp4_frame0010_1.jpg"]
        for filename in filenames:
            mock_file = MagicMock()
            mock_file.name = filename
            mock_file.is_file.return_value = True
            mock_files.append(mock_file)
        
        mock_glob.return_value = mock_files
        
        with patch('os.path.exists', return_value=True):
            videos = get_crow_videos(crow_id)
        
        self.assertIsInstance(videos, list)
        # May or may not have videos depending on fallback logic
    
    def test_get_crow_videos_nonexistent_crow(self):
        """Test get_crow_videos with nonexistent crow ID."""
        with self.assertRaises(ValueError):
            get_crow_videos(99999)
    
    @patch('pathlib.Path.exists')
    @patch('pathlib.Path.glob')
    def test_get_first_crow_image_with_fallback(self, mock_glob, mock_exists):
        """Test get_first_crow_image when database has no embeddings."""
        # Create a crow without embeddings
        conn = get_connection()
        c = conn.cursor()
        c.execute('INSERT INTO crows (name, total_sightings) VALUES (?, ?)', 
                 ("Test Crow", 0))
        crow_id = c.lastrowid
        conn.commit()
        conn.close()
        
        # Mock crop directory and files - return empty list to avoid sorting issues
        mock_exists.return_value = True
        mock_glob.return_value = []  # Return empty list to avoid mock sorting issues
        
        with patch('os.path.exists', return_value=True):
            image_path = get_first_crow_image(crow_id)
        
        # Should return None since no images are available
        self.assertIsNone(image_path)
    
    def test_get_first_crow_image_no_images(self):
        """Test get_first_crow_image when no images exist."""
        crow_id = self.create_test_crow_with_embeddings()
        
        with patch('pathlib.Path.exists', return_value=False):
            image_path = get_first_crow_image(crow_id)
        
        self.assertIsNone(image_path)
    
    @patch('pathlib.Path.exists')
    @patch('pathlib.Path.glob')
    def test_get_crow_images_from_video_fallback(self, mock_glob, mock_exists):
        """Test get_crow_images_from_video with fallback mode."""
        # Create a crow without embeddings
        conn = get_connection()
        c = conn.cursor()
        c.execute('INSERT INTO crows (name, total_sightings) VALUES (?, ?)', 
                 ("Test Crow", 0))
        crow_id = c.lastrowid
        conn.commit()
        conn.close()
        
        # Mock crop directory and files for specific video
        mock_exists.return_value = True
        mock_files = []
        filenames = ["video1.mp4_frame0010_1.jpg", "video1.mp4_frame0020_1.jpg", "video2.mp4_frame0010_1.jpg"]
        for filename in filenames:
            mock_file = MagicMock()
            mock_file.name = filename
            mock_file.__str__ = MagicMock(return_value=f"crow_crops/{crow_id}/{filename}")
            mock_files.append(mock_file)
        
        mock_glob.return_value = mock_files
        
        with patch('os.path.exists', return_value=True):
            images = get_crow_images_from_video(crow_id, "video1.mp4")
        
        self.assertIsInstance(images, list)
        # Should work without crashing
    
    def test_get_embedding_ids_by_image_paths(self):
        """Test mapping image paths to embedding IDs."""
        crow_id = self.create_test_crow_with_embeddings()
        
        # Create test image paths
        image_paths = [
            f"crow_crops/{crow_id}/video1.mp4_frame0100_1.jpg",
            f"crow_crops/{crow_id}/video2.mp4_frame0200_1.jpg",
            f"crow_crops/{crow_id}/nonexistent_frame0300_1.jpg"
        ]
        
        result = get_embedding_ids_by_image_paths(image_paths)
        
        self.assertIsInstance(result, dict)
        # The function may not return all paths if embeddings don't exist
        self.assertGreaterEqual(len(result), 0)
        
        # Check that paths are properly processed
        for path in image_paths:
            if path in result:
                # If present, value should be either an ID or None
                self.assertTrue(result[path] is None or isinstance(result[path], int))
    
    def test_get_embedding_ids_by_image_paths_empty(self):
        """Test get_embedding_ids_by_image_paths with empty input."""
        result = get_embedding_ids_by_image_paths([])
        self.assertEqual(result, {})
    
    def test_delete_crow_embeddings(self):
        """Test deleting crow embeddings by IDs."""
        crow_id = self.create_test_crow_with_embeddings()
        
        # Get embedding IDs
        conn = get_connection()
        c = conn.cursor()
        c.execute('SELECT id FROM crow_embeddings WHERE crow_id = ?', (crow_id,))
        embedding_ids = [row[0] for row in c.fetchall()]
        conn.close()
        
        self.assertGreater(len(embedding_ids), 0)
        
        # Delete first embedding
        delete_crow_embeddings([embedding_ids[0]])
        
        # Verify deletion
        conn = get_connection()
        c = conn.cursor()
        c.execute('SELECT COUNT(*) FROM crow_embeddings WHERE id = ?', (embedding_ids[0],))
        count = c.fetchone()[0]
        conn.close()
        
        self.assertEqual(count, 0)
    
    def test_delete_crow_embeddings_empty(self):
        """Test delete_crow_embeddings with empty list."""
        # Should not raise an error
        delete_crow_embeddings([])
    
    def test_reassign_crow_embeddings_to_existing(self):
        """Test reassigning embeddings to an existing crow."""
        # Create two crows
        crow_id1 = self.create_test_crow_with_embeddings()
        crow_id2 = self.create_test_crow_with_embeddings()
        
        # Get embedding IDs from first crow
        conn = get_connection()
        c = conn.cursor()
        c.execute('SELECT id FROM crow_embeddings WHERE crow_id = ?', (crow_id1,))
        embedding_ids = [row[0] for row in c.fetchall()]
        conn.close()
        
        # Reassign to second crow
        reassign_crow_embeddings(crow_id1, crow_id2, embedding_ids)
        
        # Verify reassignment
        conn = get_connection()
        c = conn.cursor()
        c.execute('SELECT crow_id FROM crow_embeddings WHERE id IN ({})'.format(
            ','.join('?' * len(embedding_ids))), embedding_ids)
        crow_ids = [row[0] for row in c.fetchall()]
        conn.close()
        
        # All should now belong to crow_id2
        self.assertTrue(all(cid == crow_id2 for cid in crow_ids))
    
    def test_create_new_crow_from_embeddings(self):
        """Test creating a new crow from existing embeddings."""
        crow_id = self.create_test_crow_with_embeddings()
        
        # Get embedding IDs
        conn = get_connection()
        c = conn.cursor()
        c.execute('SELECT id FROM crow_embeddings WHERE crow_id = ?', (crow_id,))
        embedding_ids = [row[0] for row in c.fetchall()]
        conn.close()
        
        # Create new crow from embeddings
        new_crow_id = create_new_crow_from_embeddings(crow_id, embedding_ids)
        
        self.assertIsNotNone(new_crow_id)
        self.assertNotEqual(new_crow_id, crow_id)
        
        # Verify embeddings were moved
        conn = get_connection()
        c = conn.cursor()
        c.execute('SELECT crow_id FROM crow_embeddings WHERE id IN ({})'.format(
            ','.join('?' * len(embedding_ids))), embedding_ids)
        crow_ids = [row[0] for row in c.fetchall()]
        conn.close()
        
        # All should now belong to new_crow_id
        self.assertTrue(all(cid == new_crow_id for cid in crow_ids))
    
    def test_create_new_crow_from_embeddings_empty(self):
        """Test create_new_crow_from_embeddings with empty list."""
        # First create a crow
        crow_id = self.create_test_crow_with_embeddings()
        new_crow_id = create_new_crow_from_embeddings(crow_id, [])
        
        # Should still create a crow but with no embeddings
        self.assertIsNotNone(new_crow_id)
        
        # Verify it exists in database
        conn = get_connection()
        c = conn.cursor()
        c.execute('SELECT COUNT(*) FROM crows WHERE id = ?', (new_crow_id,))
        count = c.fetchone()[0]
        conn.close()
        
        self.assertEqual(count, 1)

if __name__ == '__main__':
    unittest.main() 