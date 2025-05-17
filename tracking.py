import cv2
import numpy as np
import torchvision
from torchvision.models import resnet18
import torch
from tqdm import tqdm
from collections import defaultdict
from db import save_crow_embedding, get_all_crows, get_crow_history
from sort import Sort
from scipy.spatial.distance import cdist

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

def extract_crow_image(frame, box, padding=0.3):
    """Extract and preprocess crow image for embedding computation."""
    x1, y1, x2, y2 = map(int, box)
    h, w = frame.shape[:2]
    
    # Add padding, ensuring we don't go out of frame
    pad_w = int((x2 - x1) * padding)
    pad_h = int((y2 - y1) * padding)
    x1 = max(0, x1 - pad_w)
    y1 = max(0, y1 - pad_h)
    x2 = min(w, x2 + pad_w)
    y2 = min(h, y2 + pad_h)
    
    # Extract crow image
    crow_img = frame[y1:y2, x1:x2]
    
    # Apply adaptive histogram equalization to enhance details
    if len(crow_img.shape) == 3:
        lab = cv2.cvtColor(crow_img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        lab = cv2.merge((l,a,b))
        crow_img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    # Resize maintaining aspect ratio
    target_size = 224
    h, w = crow_img.shape[:2]
    scale = target_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    crow_img = cv2.resize(crow_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Pad to square
    square_img = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    y_offset = (target_size - new_h) // 2
    x_offset = (target_size - new_w) // 2
    square_img[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = crow_img
    
    # Convert to tensor
    crow_tensor = torch.from_numpy(square_img).permute(2, 0, 1).float() / 255.0
    return crow_tensor.unsqueeze(0)  # Add batch dimension

class EnhancedTracker:
    def __init__(self, max_age=10, min_hits=2, iou_threshold=0.15, embedding_threshold=0.6):
        """
        Enhanced tracker combining SORT with visual embeddings.
        Args:
            max_age: Maximum number of frames to keep a track alive without detection
            min_hits: Minimum number of detections before a track is confirmed
            iou_threshold: IOU threshold for matching detections to tracks
            embedding_threshold: Maximum cosine distance for embedding matching
        """
        self.sort_tracker = Sort(max_age=max_age, min_hits=min_hits, iou_threshold=iou_threshold)
        self.embedding_threshold = embedding_threshold
        self.track_embeddings = {}  # track_id -> list of embeddings
        self.track_history = defaultdict(list)  # track_id -> list of boxes
        self.frame_count = 0
        self.track_ages = defaultdict(int)  # Track how long each ID has been active
    
    def update(self, frame, detections):
        """Update tracks with new detections using both IOU and embeddings."""
        self.frame_count += 1
        
        # Prepare detections for SORT
        dets = []
        det_embeddings = []
        for det in detections:
            x1, y1, x2, y2 = det['box']
            score = det['score']
            dets.append([x1, y1, x2, y2, score])
            
            # Compute embedding for this detection
            crow_tensor = extract_crow_image(frame, det['box'])
            embedding = compute_embedding(crow_tensor)
            det_embeddings.append(embedding)
        
        dets = np.array(dets) if dets else np.empty((0, 5))
        
        # Get SORT tracks
        tracks = self.sort_tracker.update(dets)
        
        # Update track embeddings and history
        updated_tracks = []
        for track in tracks:
            x1, y1, x2, y2, track_id = track
            track_id = int(track_id)
            
            # Update track age
            self.track_ages[track_id] += 1
            
            # Find matching detection
            det_idx = None
            best_iou = 0
            for i, det in enumerate(dets):
                iou = compute_iou(det[:4], track[:4])
                if iou > best_iou:
                    best_iou = iou
                    det_idx = i
            
            if det_idx is not None:
                # Update track embedding
                if track_id not in self.track_embeddings:
                    self.track_embeddings[track_id] = []
                self.track_embeddings[track_id].append(det_embeddings[det_idx])
                # Keep more embeddings for older tracks
                max_embeddings = min(10, 5 + self.track_ages[track_id] // 30)  # Increase history for stable tracks
                self.track_embeddings[track_id] = self.track_embeddings[track_id][-max_embeddings:]
            
            # Update track history
            self.track_history[track_id].append(track[:4])
            # Keep more history for older tracks
            max_history = min(20, 10 + self.track_ages[track_id] // 30)
            self.track_history[track_id] = self.track_history[track_id][-max_history:]
            
            updated_tracks.append(track)
        
        return np.array(updated_tracks) if updated_tracks else np.empty((0, 5))

def compute_iou(box1, box2):
    """Compute Intersection over Union between two bounding boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = box1_area + box2_area - intersection
    
    return intersection / union if union > 0 else 0

def assign_crow_ids(frames, detections_list, video_path=None, max_age=5, min_hits=2, iou_threshold=0.2, embedding_threshold=0.7):
    """
    Use enhanced tracking to assign consistent IDs to detected crows across frames and videos.
    Combines SORT with visual embeddings for better identity tracking.
    """
    print("[INFO] Starting enhanced crow tracking...")
    labeled_frames = []
    tracker = EnhancedTracker(
        max_age=max_age,
        min_hits=min_hits,
        iou_threshold=iou_threshold,
        embedding_threshold=embedding_threshold
    )

    # Get known crows for display
    known_crows = get_all_crows()
    crow_ids = {crow['id']: crow['id'] for crow in known_crows}

    for frame_idx, (frame, detections) in enumerate(zip(frames, detections_list)):
        # Update tracks
        tracks = tracker.update(frame, detections)
        
        # Draw bounding boxes and IDs
        frame_copy = frame.copy()
        for track in tracks:
            x1, y1, x2, y2, track_id = map(int, track)
            
            # Get crow identity from database
            if track_id in tracker.track_embeddings and tracker.track_embeddings[track_id]:
                embedding = tracker.track_embeddings[track_id][-1]  # Use most recent embedding
                db_crow_id = save_crow_embedding(
                    embedding,
                    video_path=video_path,
                    frame_number=frame_idx,
                    confidence=track[4]  # Use track confidence
                )
                
                # Use crow ID for display
                crow_id = crow_ids.get(db_crow_id, db_crow_id)
                
                # Draw box and label
                cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Add crow ID
                label = f"Crow {crow_id}"
                cv2.putText(frame_copy, label, (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                
                # Add sighting info if available
                history = get_crow_history(db_crow_id)
                if history:
                    info = f"Seen in {history['video_count']} videos, {history['total_sightings']} times"
                    cv2.putText(frame_copy, info, (x1, y2+20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

        labeled_frames.append(frame_copy)

    print("[INFO] Enhanced tracking complete.")
    return labeled_frames
