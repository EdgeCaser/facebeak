import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import os
from pathlib import Path
from db import add_image_label, get_image_label

def label_cub_coco_images(base_dir="dataset/not_crow"):
    base_path = Path(base_dir)
    count_labeled = 0
    count_skipped = 0

    for root, dirs, files in os.walk(base_path):
        for fname in files:
            if fname.lower().endswith((".jpg", ".jpeg", ".png")) and (fname.startswith("coco_") or fname.startswith("cub_")):
                img_path = Path(root) / fname
                img_path_posix = img_path.as_posix()
                label_info = get_image_label(img_path_posix)
                if label_info and label_info['label'] == "not_a_crow":
                    count_skipped += 1
                    continue
                # Label as not_a_crow
                add_image_label(
                    img_path_posix,
                    "not_a_crow",
                    confidence=1.0,
                    reviewer_notes="Auto-labeled as not a crow (CUB/COCO import)",
                    is_training_data=True
                )
                count_labeled += 1

    print(f"Labeled {count_labeled} images as not_a_crow.")
    print(f"Skipped {count_skipped} images already labeled as not_a_crow.")

if __name__ == "__main__":
    label_cub_coco_images() 