#!/usr/bin/env python3

from db import get_all_labeled_images

def debug_labels():
    labels = get_all_labeled_images()
    print(f"Total labels in database: {len(labels)}")
    
    crow_crops_count = 0
    crow_crops2_count = 0
    other_count = 0
    
    crow_crops2_samples = []
    
    for label in labels:
        path = label['image_path']
        if 'crow_crops2' in path:
            crow_crops2_count += 1
            if len(crow_crops2_samples) < 5:
                crow_crops2_samples.append(path)
        elif 'crow_crops' in path:
            crow_crops_count += 1
        else:
            other_count += 1
    
    print(f"Labels in crow_crops: {crow_crops_count}")
    print(f"Labels in crow_crops2: {crow_crops2_count}")
    print(f"Labels in other paths: {other_count}")
    
    if crow_crops2_samples:
        print("\nSample crow_crops2 paths:")
        for path in crow_crops2_samples:
            print(f"  {path}")
    else:
        print("\nNo crow_crops2 paths found!")

if __name__ == "__main__":
    debug_labels() 