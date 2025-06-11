import os
import tarfile
import random
import shutil
import requests
from tqdm import tqdm
from PIL import Image

# ========== CONFIGURATION ==========
CUB_URL = "http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz"
CUB_TGZ = "CUB_200_2011.tgz"
CUB_DIR = "CUB_200_2011"
CUB_TARGET_SPECIES = [
    "047.American_Goldfinch",
    "017.Cardinal",
    "073.Blue_Jay",
    "087.Mallard",
    "116.Chipping_Sparrow",
    "056.Pine_Grosbeak",
    "189.Red_bellied_Woodpecker",
    "129.Song_Sparrow",
    "136.Barn_Swallow",
    "036.Northern_Flicker",
    "192.Downy_Woodpecker",
    "088.Western_Meadowlark",
    "118.House_Sparrow",
    "094.White_breasted_Nuthatch",
    "186.Cedar_Waxwing"
]
CUB_IMAGES_PER_SPECIES = 60

COCO_URL = "http://images.cocodataset.org/zips/val2017.zip"
COCO_ANN_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
COCO_ZIP = "val2017.zip"
COCO_ANN_ZIP = "annotations_trainval2017.zip"
COCO_DIR = "val2017"
COCO_ANN_DIR = "annotations"
COCO_TARGET_CATEGORIES = [
    "car", "dog", "cat", "person", "bicycle", "bus", "chair", "couch", "tv", "laptop"
]
COCO_IMAGES_PER_CATEGORY = 60

OUTPUT_DIR = "not_crow_samples"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def download_file(url, dest):
    if os.path.exists(dest):
        print(f"{dest} already exists, skipping download.")
        return
    print(f"Downloading {url} ...")
    r = requests.get(url, stream=True)
    total = int(r.headers.get('content-length', 0))
    with open(dest, 'wb') as file, tqdm(
        desc=dest, total=total, unit='iB', unit_scale=True
    ) as bar:
        for data in r.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)

def extract_tgz(tgz_path, extract_to):
    print(f"Extracting {tgz_path} ...")
    with tarfile.open(tgz_path, "r:gz") as tar:
        tar.extractall(path=extract_to)

def extract_zip(zip_path, extract_to):
    import zipfile
    print(f"Extracting {zip_path} ...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

def verify_image(path):
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False

# ========== CUB-200-2011 PROCESSING ==========
def process_cub():
    download_file(CUB_URL, CUB_TGZ)
    if not os.path.exists(CUB_DIR):
        extract_tgz(CUB_TGZ, ".")
    # Parse classes.txt to get species mapping
    class_map = {}
    with open(os.path.join(CUB_DIR, "classes.txt")) as f:
        for line in f:
            idx, name = line.strip().split(" ", 1)
            class_map[name] = int(idx)
    # Parse images.txt to get image paths
    image_map = {}
    with open(os.path.join(CUB_DIR, "images.txt")) as f:
        for line in f:
            idx, path = line.strip().split(" ", 1)
            image_map[int(idx)] = path
    # Parse image_class_labels.txt to map image idx to class idx
    img_to_class = {}
    with open(os.path.join(CUB_DIR, "image_class_labels.txt")) as f:
        for line in f:
            img_idx, class_idx = map(int, line.strip().split())
            img_to_class[img_idx] = class_idx
    # For each target species, sample images
    for species in CUB_TARGET_SPECIES:
        class_idx = class_map[species]
        img_indices = [idx for idx, cidx in img_to_class.items() if cidx == class_idx]
        random.shuffle(img_indices)
        selected = img_indices[:CUB_IMAGES_PER_SPECIES]
        for img_idx in selected:
            rel_path = image_map[img_idx]
            src = os.path.join(CUB_DIR, "images", rel_path)
            if verify_image(src):
                dst = os.path.join(OUTPUT_DIR, f"cub_{species.replace(' ', '_')}_{os.path.basename(src)}")
                shutil.copy(src, dst)

# ========== COCO PROCESSING ==========
def process_coco():
    download_file(COCO_URL, COCO_ZIP)
    download_file(COCO_ANN_URL, COCO_ANN_ZIP)
    if not os.path.exists(COCO_DIR):
        extract_zip(COCO_ZIP, ".")
    if not os.path.exists(COCO_ANN_DIR):
        extract_zip(COCO_ANN_ZIP, ".")
    
    # Load COCO annotations
    from pycocotools.coco import COCO
    ann_file = os.path.join(COCO_ANN_DIR, "instances_val2017.json")
    coco = COCO(ann_file)
    
    # Get category IDs for target categories
    cat_ids = coco.getCatIds(catNms=COCO_TARGET_CATEGORIES)
    print(f"Found categories: {COCO_TARGET_CATEGORIES}")
    print(f"Category IDs: {cat_ids}")
    
    # For each category, get all annotations and extract object crops
    total_crops = 0
    for cat_name, cat_id in zip(COCO_TARGET_CATEGORIES, cat_ids):
        print(f"\nProcessing category: {cat_name} (ID: {cat_id})")
        
        # Get all annotations for this category
        ann_ids = coco.getAnnIds(catIds=[cat_id])
        annotations = coco.loadAnns(ann_ids)
        
        print(f"Found {len(annotations)} annotations for {cat_name}")
        
        # Group annotations by image
        img_to_anns = {}
        for ann in annotations:
            img_id = ann['image_id']
            if img_id not in img_to_anns:
                img_to_anns[img_id] = []
            img_to_anns[img_id].append(ann)
        
        # Process each image
        crops_per_category = 0
        for img_id, anns in img_to_anns.items():
            if crops_per_category >= COCO_IMAGES_PER_CATEGORY:
                break
                
            # Load image info
            img_info = coco.loadImgs([img_id])[0]
            img_path = os.path.join(COCO_DIR, img_info['file_name'])
            
            if not os.path.exists(img_path):
                continue
                
            # Load image
            try:
                img = Image.open(img_path).convert('RGB')
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
                continue
            
            # Extract crops for each annotation in this image
            for ann in anns:
                if crops_per_category >= COCO_IMAGES_PER_CATEGORY:
                    break
                    
                # Get bounding box
                bbox = ann['bbox']  # [x, y, width, height]
                x, y, w, h = bbox
                
                # Skip very small objects
                if w < 32 or h < 32:
                    continue
                
                # Crop the object
                try:
                    crop = img.crop((x, y, x + w, y + h))
                    
                    # Skip very small crops
                    if crop.size[0] < 32 or crop.size[1] < 32:
                        continue
                    
                    # Save the crop
                    crop_filename = f"coco_{cat_name}_{img_id}_{ann['id']}.jpg"
                    crop_path = os.path.join(OUTPUT_DIR, crop_filename)
                    crop.save(crop_path)
                    
                    crops_per_category += 1
                    total_crops += 1
                    
                except Exception as e:
                    print(f"Error cropping annotation {ann['id']}: {e}")
                    continue
        
        print(f"Extracted {crops_per_category} crops for {cat_name}")
    
    print(f"\nTotal COCO object crops extracted: {total_crops}")

# ========== MAIN ==========
if __name__ == "__main__":
    print("Processing CUB-200-2011...")
    process_cub()
    print("Processing COCO val2017...")
    process_coco()
    print(f"Done! Images saved in {OUTPUT_DIR}/") 