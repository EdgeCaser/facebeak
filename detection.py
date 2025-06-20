import torch
import torchvision
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO
import cv2
from multi_view import create_multi_view_extractor
import signal
from contextlib import contextmanager
import functools
import json # Added import
import os # Added import
from pathlib import Path # Added import
import logging
import threading
import time
from functools import wraps
import platform # Added for the new timeout decorator
# signal is already imported globally

# Load configuration at the start of the script
CONFIG = {}
try:
    with open("config.json", "r") as f:
        CONFIG = json.load(f)
except FileNotFoundError:
    print("WARNING: config.json not found in detection.py. Using default model paths.")
except json.JSONDecodeError:
    print("WARNING: Error decoding config.json in detection.py. Using default model paths.")

# Set up logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# Custom exception class for timeouts, consistent with tracking.py
class TimeoutException(Exception): # Renamed from TimeoutError to TimeoutException
    pass

# --- PATCH: Platform check for signal.SIGALRM (copied from tracking.py) ---
IS_WINDOWS = platform.system() == 'Windows'

# Load models once at module level
print("[INFO] Loading detection models...")

# Determine model directory
model_dir_str = CONFIG.get('model_dir')
if not model_dir_str: # Handles None or empty string
    model_dir = Path('.') # Default to current directory Path object
else:
    model_dir = Path(model_dir_str)

# YOLOv8 model for first pass
yolo_model_path_obj = model_dir / 'yolov8s.pt'
try:
    yolo_model = YOLO(str(yolo_model_path_obj))  # YOLO might expect a string path
except Exception as e:
    logger.error(f"Failed to load YOLO model from {str(yolo_model_path_obj)}: {e}. Attempting to load from default path 'yolov8s.pt'.")
    yolo_model = YOLO(str(Path('.') / 'yolov8s.pt')) # Fallback to default

if torch.cuda.is_available():
    yolo_model.to('cuda')
    print("[INFO] YOLOv8 model loaded on GPU")
else:
    print("[INFO] YOLOv8 model loaded on CPU")

# Faster R-CNN model for refinement
# Note: Faster R-CNN weights are typically downloaded by torchvision and not loaded from a local file by default.
# If a specific pre-trained Faster R-CNN file was intended to be loaded from model_dir,
# the loading mechanism here would need to change significantly.
# For now, we assume the default torchvision behavior for weights.
# If you have a .pth file for faster_rcnn, the loading code would be different, e.g.:
# faster_rcnn_model_path = model_dir / 'faster_rcnn_model.pth'
# faster_rcnn_model.load_state_dict(torch.load(str(faster_rcnn_model_path)))

faster_rcnn_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
    weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
)
faster_rcnn_model.eval()

# Configure more aggressive NMS threshold to reduce overlapping bounding boxes
faster_rcnn_model.roi_heads.nms_thresh = 0.1  # Very low threshold to let our custom NMS handle merging
faster_rcnn_model.roi_heads.score_thresh = 0.2  # Lower threshold to catch more detections before our own filtering

if torch.cuda.is_available():
    faster_rcnn_model = faster_rcnn_model.cuda()
    print("[INFO] Faster R-CNN model loaded on GPU")
else:
    print("[INFO] Faster R-CNN model loaded on CPU")

# YOLO class IDs for birds
YOLO_BIRD_CLASS_ID = 14  # bird
YOLO_AIRPLANE_CLASS_ID = 4  # airplane (sometimes misclassifies birds)

# COCO class IDs for Faster R-CNN
COCO_BIRD_CLASS_ID = 16  # bird
COCO_CROW_CLASS_ID = 20  # This might not exist in COCO, but keeping for future

def extract_roi(frame, bbox, padding=0.1):
    """Extract region of interest from frame with padding."""
    x1, y1, x2, y2 = map(int, bbox)
    h, w = frame.shape[:2]
    
    # Add padding
    pad_w = int((x2 - x1) * padding)
    pad_h = int((y2 - y1) * padding)
    x1 = max(0, x1 - pad_w)
    y1 = max(0, y1 - pad_h)
    x2 = min(w, x2 + pad_w)
    y2 = min(h, y2 + pad_h)
    
    return frame[y1:y2, x1:x2], (x1, y1, x2, y2)

def compute_iou(bbox1, bbox2):
    """Compute Intersection over Union between two bounding boxes."""
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union = bbox1_area + bbox2_area - intersection
    
    return intersection / union if union > 0 else 0

def has_overlapping_crows(detections, iou_thresh=0.4):
    """
    Check if multiple crow detections overlap significantly, indicating multiple crows in frame.
    
    Args:
        detections: List of detection dictionaries with 'bbox' keys
        iou_thresh: IoU threshold above which detections are considered overlapping
        
    Returns:
        bool: True if any two detections overlap above threshold
    """
    if not detections or len(detections) < 2:
        return False
        
    for i, det1 in enumerate(detections):
        for j, det2 in enumerate(detections):
            if i >= j:  # Avoid duplicate comparisons and self-comparison
                continue
            if compute_iou(det1['bbox'], det2['bbox']) > iou_thresh:
                return True
    return False

def merge_overlapping_detections(detections, iou_threshold=0.5):
    """
    Merge overlapping detections with improved consistency.
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
        scores = [det1['score']]
        views = [det1.get('view', 'single')]
        boxes = [det1['bbox']]
        
        # Find all overlapping detections
        for j, det2 in enumerate(sorted_dets[i+1:], start=i+1):
            if j in used:
                continue
                
            iou = compute_iou(det1['bbox'], det2['bbox'])
            # Only merge if IOU exceeds threshold AND classes are the same
            # Don't merge different classes to preserve class diversity
            if iou >= iou_threshold and det1['class'] == det2['class']:
                current_group.append(det2)
                used.add(j)
                scores.append(det2['score'])
                views.append(det2.get('view', 'single'))
                boxes.append(det2['bbox'])
        
        if len(current_group) > 1:
            # Calculate view diversity bonus
            unique_views = len(set(views))
            view_bonus = 0.15 * (unique_views - 1) if unique_views > 1 else 0
            
            # Calculate merged box using union of all boxes (encompasses entire crow)
            # This ensures no part of the crow is missed regardless of confidence distribution
            x1 = min(box[0] for box in boxes)  # Leftmost edge
            y1 = min(box[1] for box in boxes)  # Topmost edge
            x2 = max(box[2] for box in boxes)  # Rightmost edge
            y2 = max(box[3] for box in boxes)  # Bottommost edge
            merged_box = [x1, y1, x2, y2]
            
            # Calculate final score
            base_score = max(scores)  # Use highest confidence as base
            confidence_bonus = 0.1 * (len(current_group) - 1)  # Bonus for multiple detections
            final_score = min(base_score + view_bonus + confidence_bonus, 1.0)  # Cap at 1.0
            
            merged.append({
                'bbox': merged_box,  # Already a list from union calculation
                'score': float(final_score),  # Ensure float type
                'class': 'crow' if any(d['class'] == 'crow' for d in current_group) else 'bird',
                'model': 'merged',
                'merged_scores': scores,
                'views': views
            })
        else:
            # For single detections, ensure consistent types
            merged.append({
                'bbox': det1['bbox'] if isinstance(det1['bbox'], list) else det1['bbox'].tolist(),
                'score': float(det1['score']),
                'class': det1['class'],
                'model': det1['model'],
                'view': det1.get('view', 'single')
            })
    
    return merged

def timeout(seconds):
    """Decorator for function timeout (copied from tracking.py)."""
    def decorator(func):
        @functools.wraps(func) # functools is already imported
        def wrapper(*args, **kwargs):
            if IS_WINDOWS:
                # Windows does not support SIGALRM, execute directly
                # For Windows, a thread-based timeout might still be possible but more complex to interrupt.
                # The original threading-based timeout in detection.py could be used here for Windows if desired,
                # but for consistency with tracking.py's approach, we'll skip signal-based timeout on Windows.
                # logger.warning(f"Timeout decorator is not using signal-based timeout on Windows for {func.__name__}.")
                return func(*args, **kwargs) 
            
            # For non-Windows, use signal-based timeout
            def handler(signum, frame):
                raise TimeoutException(f"Function {func.__name__} timed out after {seconds} seconds")
            
            original_handler = signal.signal(signal.SIGALRM, handler)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0) # Reset alarm
                signal.signal(signal.SIGALRM, original_handler) # Restore original handler
            return result
        return wrapper
    return decorator

@timeout(30)  # 30 second timeout for model inference
def _run_model_inference(model, *args, **kwargs):
    return model(*args, **kwargs)

def detect_crows_parallel(
    frames,
    score_threshold=0.3,
    yolo_threshold=0.2,
    multi_view_yolo=False,
    multi_view_rcnn=False,
    multi_view_params=None,
    nms_threshold=0.3
):
    print("[DEBUG] Entered detect_crows_parallel")
    print(f"[DEBUG] detect_crows_parallel: {len(frames)} frames, score_threshold={score_threshold}, yolo_threshold={yolo_threshold}")
    print(f"[DEBUG] NMS threshold: {nms_threshold}, Multi-view: YOLO={multi_view_yolo}, RCNN={multi_view_rcnn}")
    print(f"[DEBUG] Faster R-CNN NMS: {faster_rcnn_model.roi_heads.nms_thresh}, Score: {faster_rcnn_model.roi_heads.score_thresh}")
    detections = []
    print("[DEBUG] Setting up multi-view extractor if needed...")
    # Set up multi-view extractor if needed
    if multi_view_yolo or multi_view_rcnn:
        extractor = create_multi_view_extractor(**(multi_view_params or {}))
    else:
        extractor = None
    print("[DEBUG] Entering torch.no_grad() context...")
    with torch.no_grad():
        for idx, frame in enumerate(frames):
            print(f"[DEBUG] Processing frame {idx}")
            yolo_dets = []
            rcnn_dets = []
            try:
                # --- YOLO detections ---
                print(f"[DEBUG] Frame {idx}: Before YOLO detection")
                if multi_view_yolo and extractor is not None:
                    yolo_views = extractor.extract(frame)
                    for view in yolo_views:
                        try:
                            print(f"[DEBUG] Frame {idx}: Before YOLO model inference (multi-view)")
                            yolo_results = _run_model_inference(yolo_model, view.copy(), conf=yolo_threshold, iou=0.1)[0]  # Very low NMS threshold
                            print(f"[DEBUG] Frame {idx}: After YOLO model inference (multi-view)")
                            for bbox, score, cls in zip(yolo_results.boxes.xyxy, yolo_results.boxes.conf, yolo_results.boxes.cls):
                                print(f"[DEBUG] YOLO bbox: {bbox}, score: {score}, cls: {cls}")
                                if int(cls) == YOLO_BIRD_CLASS_ID or int(cls) == YOLO_AIRPLANE_CLASS_ID:
                                    yolo_dets.append({
                                        'bbox': bbox.cpu().numpy().tolist(),  # Ensure it's a list
                                        'score': float(score.cpu().numpy()),  # Ensure it's a float
                                        'class': 'bird',
                                        'model': 'yolo',
                                        'view': 'multi'  # Mark as multi-view
                                    })
                        except TimeoutException: # Updated to TimeoutException
                            logger.error("YOLO model inference timed out for multi-view")
                            continue  # Continue with next view instead of breaking
                else:
                    try:
                        print(f"[DEBUG] Frame {idx}: Before YOLO model inference (single-view)")
                        yolo_results = _run_model_inference(yolo_model, frame.copy(), conf=yolo_threshold, iou=0.1)[0]  # Very low NMS threshold
                        print(f"[DEBUG] Frame {idx}: After YOLO model inference (single-view)")
                        print("[DEBUG] YOLO results:", yolo_results)
                        for bbox, score, cls in zip(yolo_results.boxes.xyxy, yolo_results.boxes.conf, yolo_results.boxes.cls):
                            print(f"[DEBUG] YOLO bbox: {bbox}, score: {score}, cls: {cls}")
                            if int(cls) == YOLO_BIRD_CLASS_ID or int(cls) == YOLO_AIRPLANE_CLASS_ID:
                                yolo_dets.append({
                                    'bbox': bbox.cpu().numpy().tolist(),  # Ensure it's a list
                                    'score': float(score.cpu().numpy()),  # Ensure it's a float
                                    'class': 'bird',
                                    'model': 'yolo',
                                    'view': 'single'
                                })
                    except TimeoutException: # Updated to TimeoutException
                        logger.error("YOLO model inference timed out")
                        yolo_results = None
                print(f"[DEBUG] Frame {idx}: After YOLO detection")

                # --- Faster R-CNN detections ---
                print(f"[DEBUG] Frame {idx}: Before RCNN detection")
                if multi_view_rcnn and extractor is not None:
                    rcnn_views = extractor.extract(frame)
                    for view in rcnn_views:
                        try:
                            print(f"[DEBUG] Frame {idx}: Before RCNN model inference (multi-view)")
                            frame_tensor = torch.from_numpy(view.copy()).permute(2, 0, 1).float() / 255.0  # Make a copy of the view
                            if torch.cuda.is_available():
                                frame_tensor = frame_tensor.cuda()
                            rcnn_results = _run_model_inference(faster_rcnn_model, [frame_tensor])[0]
                            print(f"[DEBUG] Frame {idx}: After RCNN model inference (multi-view)")
                            for bbox, label, score in zip(rcnn_results['boxes'], rcnn_results['labels'], rcnn_results['scores']):
                                print(f"[DEBUG] RCNN bbox: {bbox}, label: {label}, score: {score}")
                                if (label == COCO_BIRD_CLASS_ID or label == COCO_CROW_CLASS_ID) and score > score_threshold:
                                    rcnn_dets.append({
                                        'bbox': bbox.cpu().numpy().tolist(),  # Ensure it's a list
                                        'score': float(score.cpu().numpy()),  # Ensure it's a float
                                        'class': 'crow' if label == COCO_CROW_CLASS_ID else 'bird',
                                        'model': 'rcnn',
                                        'view': 'multi'  # Mark as multi-view
                                    })
                        except TimeoutException: # Updated to TimeoutException
                            logger.error("RCNN model inference timed out for multi-view")
                            continue  # Continue with next view instead of breaking
                else:
                    try:
                        print(f"[DEBUG] Frame {idx}: Before RCNN model inference (single-view)")
                        frame_tensor = torch.from_numpy(frame.copy()).permute(2, 0, 1).float() / 255.0  # Make a copy of the frame
                        if torch.cuda.is_available():
                            frame_tensor = frame_tensor.cuda()
                        print(f"[DEBUG] Frame {idx}: Running RCNN model...")
                        rcnn_results = _run_model_inference(faster_rcnn_model, [frame_tensor])[0]
                        print(f"[DEBUG] Frame {idx}: After RCNN model inference (single-view)")
                        print("[DEBUG] RCNN results:", rcnn_results)
                        for bbox, label, score in zip(rcnn_results['boxes'], rcnn_results['labels'], rcnn_results['scores']):
                            print(f"[DEBUG] RCNN bbox: {bbox}, label: {label}, score: {score}")
                            if (label == COCO_BIRD_CLASS_ID or label == COCO_CROW_CLASS_ID) and score > score_threshold:
                                rcnn_dets.append({
                                    'bbox': bbox.cpu().numpy().tolist(),  # Ensure it's a list
                                    'score': float(score.cpu().numpy()),  # Ensure it's a float
                                    'class': 'crow' if label == COCO_CROW_CLASS_ID else 'bird',
                                    'model': 'rcnn',
                                    'view': 'single'  # Mark as single-view
                                })
                    except TimeoutException: # Updated to TimeoutException
                        logger.error("RCNN model inference timed out")
                        rcnn_results = None
                print(f"[DEBUG] Frame {idx}: After RCNN detection")

            except Exception as e:
                print(f"[DEBUG] Frame {idx}: Exception occurred: {e}")
                logger.error(f"Error processing frame {idx}: {str(e)}", exc_info=True)
            finally:
                all_dets = yolo_dets + rcnn_dets
                if all_dets:
                    # Debug: Print detections before merging
                    print(f"[DEBUG] Frame {idx}: {len(all_dets)} detections before merging:")
                    for i, det in enumerate(all_dets):
                        print(f"  Detection {i+1}: score={det['score']:.3f}, model={det['model']}, bbox={det['bbox']}")
                    
                    # Check for overlapping crows before merging
                    has_multiple_crows = has_overlapping_crows(all_dets, iou_thresh=0.4)
                    
                    # More aggressive merging: lower threshold means more boxes get merged
                    merged = merge_overlapping_detections(all_dets, iou_threshold=nms_threshold)
                    
                    # Debug: Print detections after merging
                    print(f"[DEBUG] Frame {idx}: {len(merged)} detections after merging (threshold={nms_threshold}):")
                    for i, det in enumerate(merged):
                        print(f"  Merged {i+1}: score={det['score']:.3f}, model={det['model']}, bbox={det['bbox']}")
                    
                    filtered = [d for d in merged if d['score'] >= score_threshold and d['class'] in ('bird', 'crow')]
                    
                    # Add multi-crow flag to each detection
                    for det in filtered:
                        det['multi_crow_frame'] = has_multiple_crows
                    
                    detections.append(filtered)
                else:
                    detections.append([])
                
                # --- CUDA Cache Clearing ---
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    logger.debug(f"Frame {idx}: Cleared CUDA cache.")

    print(f"[DEBUG] Finished detect_crows_parallel, detections: {detections}")
    return detections

# For backward compatibility
def detect_crows_legacy(frames, score_threshold=0.3):
    """Legacy wrapper function that uses parallel detection.
    
    This function is maintained for backward compatibility.
    New code should use detect_crows_parallel directly.
    """
    return detect_crows_parallel(frames, score_threshold=score_threshold)

# Keep cascade detection for reference but mark as deprecated
def detect_crows_cascade(frames, score_threshold=0.5, yolo_threshold=0.3):
    """Deprecated: Use detect_crows_parallel instead."""
    import warnings
    warnings.warn("Cascade detection is deprecated. Use detect_crows_parallel instead.", DeprecationWarning)
    # ... rest of cascade implementation ...

def detect_crows(*args, **kwargs):
    """Deprecated function. Use detect_crows_parallel instead."""
    return detect_crows_parallel(*args, **kwargs)
