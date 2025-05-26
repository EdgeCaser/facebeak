import unittest
import os
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
import numpy as np

# Import modules
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from db import initialize_database, save_crow_embedding, get_connection
import utilities.sync_database

class TestSuspectLineupIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        cls.test_dir = tempfile.mkdtemp()
        cls.crow_crops_dir = os.path.join(cls.test_dir, "crow_crops")
        os.makedirs(cls.crow_crops_dir, exist_ok=True)
        
    def setUp(self):
        """Set up each test with a fresh database and crop structure."""
        self.test_db_path = os.path.join(self.test_dir, f"test_db_{id(self)}.db")
        Path(self.test_db_path).touch()
        
        # Patch the database path
        self.db_patcher = patch('db.DB_PATH', Path(self.test_db_path))
        self.db_patcher.start()
        
        # Also patch sync_database's reference to db
        self.sync_db_patcher = patch('sync_database.db.DB_PATH', Path(self.test_db_path))
        self.sync_db_patcher.start()
        
        # Initialize the test database
        initialize_database()
        
        # Create comprehensive test crop structure
        self.setup_comprehensive_crop_structure()
        
    def tearDown(self):
        """Clean up after each test."""
        self.db_patcher.stop()
        self.sync_db_patcher.stop()
        
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
    
    def setup_comprehensive_crop_structure(self):
        """Create a comprehensive test crop directory structure."""
        # Create multiple crow directories with various patterns
        crow_scenarios = {
            101: {  # Crow with mixed video types
                "videos": ["IMG_1001.mp4", "VID_2001.MOV", "test_video.avi"],
                "frames_per_video": [5, 3, 2]
            },
            202: {  # Crow with many sightings
                "videos": ["IMG_1002.mp4", "IMG_1003.mp4"],
                "frames_per_video": [10, 8]
            },
            303: {  # Crow with few sightings
                "videos": ["IMG_1004.mp4"],
                "frames_per_video": [2]
            }
        }
        
        for crow_id, scenario in crow_scenarios.items():
            crow_dir = os.path.join(self.crow_crops_dir, str(crow_id))
            os.makedirs(crow_dir, exist_ok=True)
            
            for video, frame_count in zip(scenario["videos"], scenario["frames_per_video"]):
                for frame in range(frame_count):
                    for detection in [1, 2]:
                        image_name = f"{video}_frame{frame:04d}_{detection}.jpg"
                        image_path = os.path.join(crow_dir, image_name)
                        # Create non-empty file to simulate real image
                        with open(image_path, 'wb') as f:
                            f.write(b'fake_jpeg_data')
    
    def test_database_sync_integration(self):
        """Test the database sync utility with real crop structure."""
        # Initially database should be empty except for schema
        conn = get_connection()
        c = conn.cursor()
        c.execute('SELECT COUNT(*) FROM crows')
        initial_count = c.fetchone()[0]
        conn.close()
        
        self.assertEqual(initial_count, 0)
        
        # Run the sync
        sync_database.sync_database_with_crops()
        
        # Verify crows were created
        conn = get_connection()
        c = conn.cursor()
        c.execute('SELECT COUNT(*) FROM crows')
        final_count = c.fetchone()[0]
        
        c.execute('SELECT id, name, total_sightings FROM crows ORDER BY id')
        crows = c.fetchall()
        conn.close()
        
        # Should have created 3 crows (101, 202, 303)
        self.assertEqual(final_count, 3)
        
        # Verify crow details
        expected_crows = {
            101: ("Crow 101", 20),  # 5*2 + 3*2 + 2*2 = 20 images
            202: ("Crow 202", 36),  # 10*2 + 8*2 = 36 images
            303: ("Crow 303", 4),   # 2*2 = 4 images
        }
        
        for crow_id, name, sightings in crows:
            self.assertIn(crow_id, expected_crows)
            expected_name, expected_sightings = expected_crows[crow_id]
            self.assertEqual(name, expected_name)
            self.assertEqual(sightings, expected_sightings)
    
    def test_end_to_end_suspect_lineup_workflow(self):
        """Test complete suspect lineup workflow."""
        from db import (
            get_all_crows, get_crow_videos, get_first_crow_image,
            get_crow_images_from_video, get_embedding_ids_by_image_paths,
            create_new_crow_from_embeddings, delete_crow_embeddings
        )
        
        # First sync the database
        sync_database.sync_database_with_crops()
        
        # Step 1: Load all crows (simulating GUI initialization)
        crows = get_all_crows()
        self.assertEqual(len(crows), 3)
        
        # Step 2: Select a crow (simulate user selection)
        selected_crow = crows[0]  # Should be crow 101
        crow_id = selected_crow['id']
        
        # Step 3: Get crow's first image
        first_image = get_first_crow_image(crow_id)
        self.assertIsNotNone(first_image)
        self.assertIn(str(crow_id), first_image)
        
        # Step 4: Get crow's videos
        videos = get_crow_videos(crow_id)
        self.assertGreater(len(videos), 0)
        
        # Step 5: Select a video and get its images
        selected_video = videos[0]
        video_path = selected_video['video_path']
        images = get_crow_images_from_video(crow_id, video_path)
        self.assertGreater(len(images), 0)
        
        # Step 6: Simulate user classifications
        classifications = {}
        for i, image_path in enumerate(images[:3]):  # First 3 images
            if i == 0:
                classifications[image_path] = "different_crow"
            elif i == 1:
                classifications[image_path] = "not_a_crow"
            else:
                classifications[image_path] = "same_crow"
        
        # Step 7: Get embedding IDs for classified images
        embedding_ids_map = get_embedding_ids_by_image_paths(list(classifications.keys()))
        
        # Step 8: Process classifications (simulate save operation)
        different_crow_images = [path for path, cls in classifications.items() if cls == "different_crow"]
        not_crow_images = [path for path, cls in classifications.items() if cls == "not_a_crow"]
        
        different_crow_embedding_ids = [embedding_ids_map[path] for path in different_crow_images if embedding_ids_map[path]]
        not_crow_embedding_ids = [embedding_ids_map[path] for path in not_crow_images if embedding_ids_map[path]]
        
        # Create new crow for "different_crow" images
        if different_crow_embedding_ids:
            new_crow_id = create_new_crow_from_embeddings(different_crow_embedding_ids)
            self.assertIsNotNone(new_crow_id)
            self.assertNotEqual(new_crow_id, crow_id)
        
        # Delete "not_a_crow" images
        if not_crow_embedding_ids:
            delete_crow_embeddings(not_crow_embedding_ids)
        
        # Step 9: Verify final state
        updated_crows = get_all_crows()
        # Should have one additional crow if any different_crow classifications were made
        if different_crow_embedding_ids:
            self.assertEqual(len(updated_crows), 4)  # Original 3 + 1 new
    
    def test_fallback_mode_integration(self):
        """Test integration when database has no embeddings (fallback mode)."""
        from db import get_crow_videos, get_first_crow_image, get_crow_images_from_video
        
        # Sync database to create crows without embeddings
        sync_database.sync_database_with_crops()
        
        # Get a crow
        conn = get_connection()
        c = conn.cursor()
        c.execute('SELECT id FROM crows LIMIT 1')
        crow_id = c.fetchone()[0]
        conn.close()
        
        # Test fallback functions work
        videos = get_crow_videos(crow_id)
        self.assertGreater(len(videos), 0)
        
        first_image = get_first_crow_image(crow_id)
        self.assertIsNotNone(first_image)
        
        # Test getting images from specific video
        video_path = videos[0]['video_path']
        images = get_crow_images_from_video(crow_id, video_path)
        self.assertGreater(len(images), 0)
        
        # All images should be from the specified video
        for image_path in images:
            video_name = Path(video_path).name
            self.assertIn(video_name.replace('.', '_'), image_path)
    
    def test_sync_database_edge_cases(self):
        """Test database sync with edge cases."""
        # Test with empty crop directory
        empty_crops_dir = os.path.join(self.test_dir, "empty_crops")
        os.makedirs(empty_crops_dir, exist_ok=True)
        
        with patch('sync_database.CROP_BASE_DIR', Path(empty_crops_dir)):
            sync_database.sync_database_with_crops()
        
        # Should handle gracefully
        conn = get_connection()
        c = conn.cursor()
        c.execute('SELECT COUNT(*) FROM crows')
        count = c.fetchone()[0]
        conn.close()
        
        self.assertEqual(count, 0)
        
        # Test with directories that have no images
        no_images_dir = os.path.join(self.test_dir, "no_images_crops")
        os.makedirs(no_images_dir, exist_ok=True)
        os.makedirs(os.path.join(no_images_dir, "999"), exist_ok=True)  # Empty crow dir
        
        with patch('sync_database.CROP_BASE_DIR', Path(no_images_dir)):
            sync_database.sync_database_with_crops()
        
        # Should still create crow entry but with 0 sightings
        conn = get_connection()
        c = conn.cursor()
        c.execute('SELECT total_sightings FROM crows WHERE id = 999')
        result = c.fetchone()
        conn.close()
        
        if result:  # Only check if crow was created
            self.assertEqual(result[0], 0)
    
    def test_gui_launcher_integration(self):
        """Test integration with GUI launcher."""
        from facebeak import FacebeakGUI
        
        # Mock the GUI components to test the suspect lineup launch
        root = MagicMock()
        
        with patch('tkinter.ttk.Frame'), \
             patch('tkinter.ttk.LabelFrame'), \
             patch('tkinter.Listbox'), \
             patch('tkinter.ttk.Scrollbar'), \
             patch('tkinter.ttk.Button'), \
             patch('tkinter.ttk.Entry'), \
             patch('tkinter.Text'), \
             patch('tkinter.ttk.Label'), \
             patch('tkinter.BooleanVar'), \
             patch('tkinter.ttk.Style'), \
             patch('subprocess.Popen') as mock_popen:
            
            app = FacebeakGUI(root)
            
            # Test launching suspect lineup
            app.launch_suspect_lineup()
            
            # Should attempt to run suspect lineup script
            mock_popen.assert_called_once()
            args = mock_popen.call_args[0][0]
            self.assertIn('suspect_lineup.py', ' '.join(args))
    
    def test_image_file_parsing_integration(self):
        """Test integration of image filename parsing logic."""
        from db import get_embedding_ids_by_image_paths
        
        # Create some test embeddings first
        embedding1 = np.random.rand(512).astype(np.float32)
        embedding2 = np.random.rand(512).astype(np.float32)
        
        crow_id1 = save_crow_embedding(embedding1, "IMG_1001.mp4", 10, 0.95)
        crow_id2 = save_crow_embedding(embedding2, "VID_2001.MOV", 20, 0.90)
        
        # Test image paths that should match
        test_paths = [
            f"crow_crops/{crow_id1}/IMG_1001.mp4_frame0010_1.jpg",
            f"crow_crops/{crow_id2}/VID_2001.MOV_frame0020_1.jpg",
            f"crow_crops/{crow_id1}/nonexistent_frame0030_1.jpg",  # No matching embedding
        ]
        
        result = get_embedding_ids_by_image_paths(test_paths)
        
        # Should map paths to embedding IDs where possible
        self.assertEqual(len(result), len(test_paths))
        
        # First two should have embedding IDs, third should be None
        self.assertIsNotNone(result[test_paths[0]])
        self.assertIsNotNone(result[test_paths[1]])
        self.assertIsNone(result[test_paths[2]])
    
    def test_error_handling_integration(self):
        """Test error handling in integrated workflows."""
        from db import get_crow_videos, get_first_crow_image
        
        # Test with nonexistent crow
        with self.assertRaises(ValueError):
            get_crow_videos(99999)
        
        # Test with missing crop directories
        with patch('pathlib.Path.exists', return_value=False):
            result = get_first_crow_image(1)
            self.assertIsNone(result)
        
        # Test sync with permission errors
        with patch('pathlib.Path.iterdir', side_effect=PermissionError):
            # Should handle gracefully without crashing
            try:
                sync_database.sync_database_with_crops()
            except PermissionError:
                self.fail("sync_database_with_crops should handle permission errors gracefully")

if __name__ == '__main__':
    unittest.main() 