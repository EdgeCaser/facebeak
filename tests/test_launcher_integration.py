#!/usr/bin/env python3
"""
Test script to verify the image ingestion button integration in the main launcher.
"""

import os
import sys
import tempfile
import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TestLauncherIntegration(unittest.TestCase):
    
    def test_image_ingestion_button_exists(self):
        """Test that the image ingestion button is properly added to the launcher."""
        from facebeak import FacebeakGUI
        import tkinter as tk
        
        # Create a temporary root window
        root = tk.Tk()
        
        try:
            # Create the GUI
            app = FacebeakGUI(root)
            
            # Check that the ingestion button exists
            self.assertTrue(hasattr(app, 'ingestion_button'), 
                          "Image ingestion button should exist")
            
            # Check that the launch method exists
            self.assertTrue(hasattr(app, 'launch_image_ingestion'), 
                          "launch_image_ingestion method should exist")
            
            # Check that the button is properly configured
            self.assertEqual(app.ingestion_button['text'], "Launch Image Ingestion",
                           "Button should have correct text")
            
        finally:
            root.destroy()
    
    def test_launch_method_path_construction(self):
        """Test that the launch method constructs the correct path."""
        from facebeak import FacebeakGUI
        import tkinter as tk
        
        # Create a temporary root window
        root = tk.Tk()
        
        try:
            # Create the GUI
            app = FacebeakGUI(root)
            
            # Test path construction
            script_dir = Path(__file__).resolve().parent.parent
            expected_path = script_dir / 'gui' / 'image_ingestion_gui.py'
            
            # Check that the expected path exists
            self.assertTrue(expected_path.exists(), 
                          f"Image ingestion script should exist at {expected_path}")
            
        finally:
            root.destroy()
    
    @patch('subprocess.Popen')
    def test_launch_method_calls_subprocess(self, mock_popen):
        """Test that the launch method calls subprocess correctly."""
        from facebeak import FacebeakGUI
        import tkinter as tk
        
        # Create a temporary root window
        root = tk.Tk()
        
        try:
            # Create the GUI
            app = FacebeakGUI(root)
            
            # Mock the subprocess call
            mock_popen.return_value = MagicMock()
            
            # Call the launch method
            app.launch_image_ingestion()
            
            # Check that subprocess.Popen was called
            mock_popen.assert_called_once()
            
            # Check that the correct arguments were passed
            call_args = mock_popen.call_args[0][0]
            self.assertEqual(len(call_args), 2, "Should have 2 arguments: python path and script path")
            self.assertTrue(call_args[1].endswith('image_ingestion_gui.py'), 
                          "Second argument should be the image ingestion script path")
            
        finally:
            root.destroy()
    
    def test_button_layout(self):
        """Test that the button is properly positioned in the layout."""
        from facebeak import FacebeakGUI
        import tkinter as tk
        
        # Create a temporary root window
        root = tk.Tk()
        
        try:
            # Create the GUI
            app = FacebeakGUI(root)
            
            # Check that all three buttons exist in the review frame
            self.assertTrue(hasattr(app, 'review_button'), "Image review button should exist")
            self.assertTrue(hasattr(app, 'ingestion_button'), "Image ingestion button should exist")
            self.assertTrue(hasattr(app, 'suspect_lineup_button'), "Suspect lineup button should exist")
            
            # Check button texts
            self.assertEqual(app.review_button['text'], "Launch Image Reviewer")
            self.assertEqual(app.ingestion_button['text'], "Launch Image Ingestion")
            self.assertEqual(app.suspect_lineup_button['text'], "Launch Suspect Lineup")
            
        finally:
            root.destroy()

def main():
    """Run the tests."""
    print("Testing launcher integration...")
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestLauncherIntegration)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    if result.wasSuccessful():
        print("\n✅ All launcher integration tests passed!")
        return 0
    else:
        print("\n❌ Some tests failed!")
        return 1

if __name__ == "__main__":
    exit(main()) 