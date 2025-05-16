import cv2
import numpy as np
import torchvision
from torchvision.models import resnet18
import torch
from tqdm import tqdm
from collections import defaultdict
from db import get_all_embeddings, save_crow_embedding, update_last_seen
from sort import Sort

# Load model once at module level
model = resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
model = torch.nn.Sequential(*list(model.children())[:-1])  # Remove classification layer
model.eval()
if torch.cuda.is_available():
    model = model.cuda()
    print("[INFO] Tracking model loaded on GPU")
else:
    print("[INFO] Tracking model loaded on CPU")

def compute_embedding(img_tensor):
    """Compute feature embedding for an image tensor."""
    with torch.no_grad():
        if torch.cuda.is_available():
            img_tensor = img_tensor.cuda()
        features = model(img_tensor)
        return features.squeeze().cpu().numpy()

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def add_padding(box, frame_shape, pad_ratio=0.1):
    x1, y1, x2, y2 = box
    h, w = frame_shape[:2]
    pad_w = int((x2 - x1) * pad_ratio)
    pad_h = int((y2 - y1) * pad_ratio)
    x1 = max(0, x1 - pad_w)
    y1 = max(0, y1 - pad_h)
    x2 = min(w, x2 + pad_w)
    y2 = min(h, y2 + pad_h)
    return np.array([x1, y1, x2, y2])

def assign_crow_ids(frames, detections_list):
    """
    Use SORT to assign consistent IDs to detected crows across frames.
    Args:
        frames: List of video frames (as numpy arrays)
        detections_list: List of detections per frame, where each detection is a dict with 'box' and 'score'
    Returns:
        List of frames with bounding boxes and consistent IDs drawn
    """
    print("[INFO] Starting crow identity tracking with SORT...")
    labeled_frames = []
    sort_tracker = Sort()  # You can adjust max_age, min_hits, iou_threshold if needed

    for frame_idx, (frame, detections) in enumerate(zip(frames, detections_list)):
        # Prepare detections for SORT: [x1, y1, x2, y2, score]
        dets = []
        for det in detections:
            x1, y1, x2, y2 = det['box']
            score = det['score']
            dets.append([x1, y1, x2, y2, score])
        dets = np.array(dets) if dets else np.empty((0, 5))

        # Update SORT tracker
        tracks = sort_tracker.update(dets)
        # tracks: [[x1, y1, x2, y2, track_id], ...]

        # Draw bounding boxes and IDs
        frame_copy = frame.copy()
        for track in tracks:
            x1, y1, x2, y2, track_id = map(int, track)
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame_copy, f"Crow {track_id}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        labeled_frames.append(frame_copy)

    print("[INFO] Crow tracking with SORT complete.")
    return labeled_frames
