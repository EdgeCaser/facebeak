#!/usr/bin/env python3

import sys
import tempfile
from pathlib import Path

# Import the fixture
from tests.conftest import video_test_data

# Create a mock tmp_path_factory
class MockTmpPathFactory:
    def mktemp(self, name):
        return Path(tempfile.mkdtemp(prefix=name))

def main():
    print("=== Debug Video Test Data Fixture ===")
    
    # Check if unit_testing_videos exists
    video_dir = Path("unit_testing_videos")
    print(f"unit_testing_videos exists: {video_dir.exists()}")
    
    if video_dir.exists():
        video_files = list(video_dir.glob("*.mp4"))
        print(f"Video files found: {len(video_files)}")
        for vf in video_files:
            print(f"  - {vf.name} ({vf.stat().st_size / 1024 / 1024:.1f} MB)")
    else:
        print("unit_testing_videos directory not found!")
        return
    
    print("\n=== Running video_test_data fixture ===")
    tmp_factory = MockTmpPathFactory()
    
    try:
        result = video_test_data(tmp_factory)
        
        print(f"Fixture result valid: {result['valid']}")
        print(f"Base directory: {result['base_dir']}")
        
        if result['valid']:
            print(f"Metadata keys: {list(result['metadata'].keys())}")
            for crow_id, data in result['metadata'].items():
                print(f"  {crow_id}: {len(data.get('images', {}))} images")
                # Check if images actually exist
                images_dir = result['base_dir'] / crow_id / 'images'
                if images_dir.exists():
                    actual_images = list(images_dir.glob("*.jpg"))
                    print(f"    Actual image files: {len(actual_images)}")
                else:
                    print(f"    Images directory doesn't exist: {images_dir}")
        else:
            print("Video processing failed - check error messages above")
            
    except Exception as e:
        print(f"Error running fixture: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 