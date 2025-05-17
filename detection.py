import torch
import torchvision
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO
import cv2

# Load models once at module level
print("[INFO] Loading detection models...")

# YOLOv8 model for first pass
yolo_model = YOLO('yolov8s.pt')  # Using small model for better accuracy while maintaining speed
if torch.cuda.is_available():
    yolo_model.to('cuda')
    print("[INFO] YOLOv8 model loaded on GPU")
else:
    print("[INFO] YOLOv8 model loaded on CPU")

# Faster R-CNN model for refinement
faster_rcnn_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
    weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
)
faster_rcnn_model.eval()
if torch.cuda.is_available():
    faster_rcnn_model = faster_rcnn_model.cuda()
    print("[INFO] Faster R-CNN model loaded on GPU")
else:
    print("[INFO] Faster R-CNN model loaded on CPU")

# COCO dataset class IDs
COCO_BIRD_CLASS_ID = 16  # General bird class
COCO_CROW_CLASS_ID = 20  # Specific crow class (if available in the model)
YOLO_BIRD_CLASS_ID = 14  # YOLO's bird class ID

def extract_roi(frame, box, padding=0.1):
    """Extract region of interest from frame with padding."""
    x1, y1, x2, y2 = map(int, box)
    h, w = frame.shape[:2]
    
    # Add padding
    pad_w = int((x2 - x1) * padding)
    pad_h = int((y2 - y1) * padding)
    x1 = max(0, x1 - pad_w)
    y1 = max(0, y1 - pad_h)
    x2 = min(w, x2 + pad_w)
    y2 = min(h, y2 + pad_h)
    
    return frame[y1:y2, x1:x2], (x1, y1, x2, y2)

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

def merge_overlapping_detections(detections, iou_threshold=0.4):
    """
    Merge overlapping detections from different models.
    Args:
        detections: List of detection dictionaries
        iou_threshold: IOU threshold for merging
    Returns:
        List of merged detections
    """
    if not detections:
        return []
    
    # Sort detections by score in descending order
    sorted_dets = sorted(detections, key=lambda x: x['score'], reverse=True)
    merged = []
    used = set()
    
    for i, det1 in enumerate(sorted_dets):
        if i in used:
            continue
            
        current_group = [det1]
        used.add(i)
        
        # Find overlapping detections
        for j, det2 in enumerate(sorted_dets):
            if j in used:
                continue
                
            iou = compute_iou(det1['box'], det2['box'])
            if iou > iou_threshold:
                current_group.append(det2)
                used.add(j)
        
        if len(current_group) > 1:
            # Merge overlapping detections
            boxes = np.array([d['box'] for d in current_group])
            scores = [d['score'] for d in current_group]
            
            # Use weighted average of boxes based on scores
            weights = np.array(scores) / sum(scores)
            merged_box = np.average(boxes, weights=weights, axis=0)
            
            merged.append({
                'box': merged_box,
                'score': max(scores),  # Use highest confidence
                'class': 'crow' if any(d['class'] == 'crow' for d in current_group) else 'bird',
                'model': 'merged',
                'merged_scores': scores  # Keep track of original scores
            })
        else:
            merged.append(det1)
    
    return merged

def detect_crows_parallel(frames, score_threshold=0.3, yolo_threshold=0.2):
    """
    Detect birds using parallel YOLOv8 and Faster R-CNN models.
    A detection is considered valid if either model detects it with sufficient confidence.
    Args:
        frames: List of video frames
        score_threshold: Minimum confidence score for final detections
        yolo_threshold: Minimum confidence score for YOLO detections
    Returns:
        List of detections per frame
    """
    detections = []
    print(f"[INFO] Processing {len(frames)} frames with parallel detection")
    
    with torch.no_grad():
        for frame in tqdm(frames, desc="Detecting birds"):
            # Run both models in parallel
            yolo_results = yolo_model(frame, conf=yolo_threshold)[0]
            frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
            if torch.cuda.is_available():
                frame_tensor = frame_tensor.cuda()
            rcnn_results = faster_rcnn_model([frame_tensor])[0]
            
            # Process YOLO detections
            yolo_dets = []
            for box, score, cls in zip(yolo_results.boxes.xyxy, yolo_results.boxes.conf, yolo_results.boxes.cls):
                if int(cls) == YOLO_BIRD_CLASS_ID:
                    yolo_dets.append({
                        'box': box.cpu().numpy(),
                        'score': score.cpu().numpy(),
                        'class': 'bird',
                        'model': 'yolo'
                    })
            
            # Process Faster R-CNN detections
            rcnn_dets = []
            for box, label, score in zip(rcnn_results['boxes'], rcnn_results['labels'], rcnn_results['scores']):
                if (label == COCO_BIRD_CLASS_ID or label == COCO_CROW_CLASS_ID) and score > score_threshold:
                    rcnn_dets.append({
                        'box': box.cpu().numpy(),
                        'score': score.cpu().numpy(),
                        'class': 'crow' if label == COCO_CROW_CLASS_ID else 'bird',
                        'model': 'rcnn'
                    })
            
            # Combine detections from both models
            all_dets = yolo_dets + rcnn_dets
            
            # Merge overlapping detections
            if all_dets:
                merged_dets = merge_overlapping_detections(all_dets, iou_threshold=0.4)
                # For merged detections, use the higher confidence score
                for det in merged_dets:
                    if 'merged_scores' in det:
                        det['score'] = max(det['merged_scores'])
                        del det['merged_scores']
                detections.append(merged_dets)
            else:
                detections.append([])
    
    total_detections = sum(len(d) for d in detections)
    frames_with_birds = sum(1 for d in detections if d)
    print(f"[INFO] Parallel detection complete. Found {total_detections} birds in {frames_with_birds} frames")
    return detections

# For backward compatibility
def detect_crows(frames, score_threshold=0.3):
    """Legacy function that now uses parallel detection."""
    return detect_crows_parallel(frames, score_threshold=score_threshold)

# Keep cascade detection for reference but mark as deprecated
def detect_crows_cascade(frames, score_threshold=0.5, yolo_threshold=0.3):
    """Deprecated: Use detect_crows_parallel instead."""
    import warnings
    warnings.warn("Cascade detection is deprecated. Use detect_crows_parallel instead.", DeprecationWarning)
    # ... rest of cascade implementation ...
