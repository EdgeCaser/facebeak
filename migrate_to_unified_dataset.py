import os
import shutil
from glob import glob

# Define source and destination directories
SRC_NON_CROW = 'not_crow_samples_cropped_512'
SRC_CROW = 'crow_crops/videos'
SRC_HARD_NEG = 'crow_crops/hard_negatives'
DST_ROOT = 'dataset'
DST_NON_CROW = os.path.join(DST_ROOT, 'not_crow')
DST_CROW = os.path.join(DST_ROOT, 'crows', 'generic')
DST_HARD_NEG = os.path.join(DST_NON_CROW, 'hard_negatives')

# Create destination directories
os.makedirs(DST_NON_CROW, exist_ok=True)
os.makedirs(DST_CROW, exist_ok=True)
os.makedirs(DST_HARD_NEG, exist_ok=True)

def move_images(src_dir, dst_dir, exts=(".jpg", ".jpeg", ".png")):
    count = 0
    for root, _, files in os.walk(src_dir):
        for f in files:
            if f.lower().endswith(exts):
                src_path = os.path.join(root, f)
                dst_path = os.path.join(dst_dir, f)
                shutil.move(src_path, dst_path)
                count += 1
    return count

# Move non-crow images
non_crow_count = move_images(SRC_NON_CROW, DST_NON_CROW)

# Move crow images
crow_count = 0
for video_dir in glob(os.path.join(SRC_CROW, '*')):
    if os.path.isdir(video_dir):
        crow_count += move_images(video_dir, DST_CROW)

# Move hard negatives
hard_neg_count = 0
for neg_dir in glob(os.path.join(SRC_HARD_NEG, '*')):
    if os.path.isdir(neg_dir):
        hard_neg_count += move_images(neg_dir, DST_HARD_NEG)

print(f"Moved {non_crow_count} non-crow images to {DST_NON_CROW}")
print(f"Moved {crow_count} crow images to {DST_CROW}")
print(f"Moved {hard_neg_count} hard negative images to {DST_HARD_NEG}") 