import db

print("Checking database contents...")
crows = db.get_all_crows()
print(f"Total crows in database: {len(crows)}")

print("\nFirst 10 crows:")
for i, crow in enumerate(crows[:10]):
    print(f"ID: {crow['id']}, Name: {crow.get('name', 'None')}, Sightings: {crow.get('total_sightings', 0)}")

if crows:
    first_crow_id = crows[0]['id']
    print(f"\nTesting first crow (ID: {first_crow_id}):")
    
    # Check what videos this crow appears in
    videos = db.get_crow_videos(first_crow_id)
    print(f"Videos for crow {first_crow_id}: {len(videos)}")
    for video in videos:
        print(f"  - {video['video_path']} ({video['sighting_count']} sightings)")
    
    # Try to get first image (should now work with fallback)
    first_image = db.get_first_crow_image(first_crow_id)
    print(f"First image for crow {first_crow_id}: {first_image}")
    
    # Check directory exists
    import os
    crop_dir = f"crow_crops/{first_crow_id}"
    print(f"Directory {crop_dir} exists: {os.path.exists(crop_dir)}")
    
    # Test video image loading
    if videos:
        test_video = videos[0]['video_path']
        images = db.get_crow_images_from_video(first_crow_id, test_video)
        print(f"Images from {test_video}: {len(images)}")
        if images:
            print(f"  First image: {images[0]}")
            
print("\nChecking available crop directories:")
import os
crop_dirs = [d for d in os.listdir("crow_crops") if os.path.isdir(f"crow_crops/{d}") and d.isdigit()]
print(f"Available crop directories: {sorted(crop_dirs)[:10]}...")  # Show first 10 