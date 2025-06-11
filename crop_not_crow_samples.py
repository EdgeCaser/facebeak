import os
from PIL import Image
import json

# ========== CONFIGURATION ==========
CUB_DIR = "CUB_200_2011"
CUB_INPUT_DIR = "not_crow_samples"
CUB_BBOX_FILE = os.path.join(CUB_DIR, "bounding_boxes.txt")
CUB_IMG_MAP_FILE = os.path.join(CUB_DIR, "images.txt")

COCO_DIR = "val2017"
COCO_ANN_FILE = "annotations/instances_val2017.json"
COCO_INPUT_DIR = "not_crow_samples"
OUTPUT_DIR = "not_crow_samples_cropped_512"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def crop_square_and_resize(img, bbox):
    """Crop to bounding box, expand to square, then resize to 512x512"""
    x, y, w, h = bbox
    x, y, w, h = int(x), int(y), int(w), int(h)
    
    # Calculate the center of the bounding box
    center_x = x + w // 2
    center_y = y + h // 2
    
    # Determine the size of the square crop (use the larger dimension)
    square_size = max(w, h)
    
    # Calculate the square crop coordinates
    square_x1 = center_x - square_size // 2
    square_y1 = center_y - square_size // 2
    square_x2 = square_x1 + square_size
    square_y2 = square_y1 + square_size
    
    # Ensure the square crop is within image bounds
    square_x1 = max(0, square_x1)
    square_y1 = max(0, square_y1)
    square_x2 = min(img.width, square_x2)
    square_y2 = min(img.height, square_y2)
    
    # Adjust if we went out of bounds
    if square_x2 - square_x1 < square_size:
        if square_x1 == 0:
            square_x2 = min(img.width, square_size)
        else:
            square_x1 = max(0, img.width - square_size)
    if square_y2 - square_y1 < square_size:
        if square_y1 == 0:
            square_y2 = min(img.height, square_size)
        else:
            square_y1 = max(0, img.height - square_size)
    
    # Crop the square from the original image
    square_crop = img.crop((square_x1, square_y1, square_x2, square_y2))
    
    # Resize to 512x512
    return square_crop.resize((512, 512), Image.Resampling.LANCZOS)

def resize_to_512(img):
    """Resize image to 512x512 by cropping from center"""
    # Calculate the center crop
    width, height = img.size
    min_side = min(width, height)
    
    # Calculate crop coordinates to get a square from the center
    left = (width - min_side) // 2
    top = (height - min_side) // 2
    right = left + min_side
    bottom = top + min_side
    
    # Crop the square from the center
    square_crop = img.crop((left, top, right, bottom))
    
    # Resize to 512x512
    return square_crop.resize((512, 512), Image.Resampling.LANCZOS)

# --- CUB Processing ---
def process_cub():
    # Build image_id to filename map
    id_to_file = {}
    with open(CUB_IMG_MAP_FILE) as f:
        for line in f:
            img_id, path = line.strip().split(" ", 1)
            id_to_file[int(img_id)] = os.path.basename(path)
    
    # Build filename to bbox map
    file_to_bbox = {}
    with open(CUB_BBOX_FILE) as f:
        for line in f:
            img_id, x, y, w, h = line.strip().split()
            fname = id_to_file[int(img_id)]
            file_to_bbox[fname] = (float(x), float(y), float(w), float(h))
    
    # Process images
    processed_count = 0
    cropped_count = 0
    resized_count = 0
    
    for fname in os.listdir(CUB_INPUT_DIR):
        if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        
        # Handle CUB filename mapping
        if fname.startswith('cub_'):
            # Convert cub_017.Cardinal_Cardinal_0001_17057.jpg to Cardinal_0001_17057.jpg
            parts = fname[4:].split('_')  # Remove 'cub_' and split
            if len(parts) >= 3:
                # Reconstruct the original CUB filename format
                species_part = parts[0]  # 017.Cardinal
                bird_name = parts[1]     # Cardinal
                number_part = parts[2]   # 0001
                id_part = parts[3].split('.')[0]  # 17057 (remove .jpg)
                lookup_name = f"{bird_name}_{number_part}_{id_part}.jpg"
            else:
                print(f"Could not parse CUB filename {fname}, skipping.")
                continue
        else:
            lookup_name = fname
            
        path = os.path.join(CUB_INPUT_DIR, fname)
        
        # Check if we have a bounding box for this image
        if lookup_name in file_to_bbox:
            # Process with bounding box
            with Image.open(path) as img:
                img = img.convert("RGB")
                cropped = crop_square_and_resize(img, file_to_bbox[lookup_name])
                cropped.save(os.path.join(OUTPUT_DIR, fname))
                cropped_count += 1
        else:
            # Process without bounding box - just resize
            with Image.open(path) as img:
                img = img.convert("RGB")
                resized = resize_to_512(img)
                resized.save(os.path.join(OUTPUT_DIR, fname))
                resized_count += 1
        
        processed_count += 1
        if processed_count % 50 == 0:
            print(f"Processed {processed_count} CUB images... (cropped: {cropped_count}, resized: {resized_count})")
    
    print(f"Successfully processed {processed_count} CUB images (cropped: {cropped_count}, resized: {resized_count}).")

# --- COCO Processing ---
def process_coco():
    print("Processing COCO images...")
    coco_input_dir = "not_crow_samples"
    processed_count = 0
    
    for fname in os.listdir(coco_input_dir):
        if not fname.startswith('coco_'):
            continue
        if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
            
        path = os.path.join(coco_input_dir, fname)
        
        # COCO images are already cropped to object bounding boxes, so just resize
        with Image.open(path) as img:
            img = img.convert("RGB")
            resized = resize_to_512(img)
            resized.save(os.path.join(OUTPUT_DIR, fname))
            processed_count += 1
            
        if processed_count % 50 == 0:
            print(f"Processed {processed_count} COCO images...")
    
    print(f"Successfully processed {processed_count} COCO images.")

if __name__ == "__main__":
    print("Cropping and resizing CUB-200-2011 images...")
    process_cub()
    print("Cropping and resizing COCO images...")
    process_coco()
    print(f"Done! Cropped images saved in {OUTPUT_DIR}/") 