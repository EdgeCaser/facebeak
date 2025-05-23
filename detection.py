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
import logging
import threading
import time
from functools import wraps

# Set up logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

class TimeoutError(Exception):
    pass

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
            # Always merge if IOU exceeds threshold
            if iou >= iou_threshold:
                current_group.append(det2)
                used.add(j)
                scores.append(det2['score'])
                views.append(det2.get('view', 'single'))
                boxes.append(det2['bbox'])
        
        if len(current_group) > 1:
            # Calculate view diversity bonus
            unique_views = len(set(views))
            view_bonus = 0.15 * (unique_views - 1) if unique_views > 1 else 0
            
            # Use weighted average of boxes with squared confidence-based weighting
            weights = np.array(scores) ** 2  # Square the weights to favor higher confidence more strongly
            weights = weights / np.sum(weights)  # Normalize weights
            
            # Calculate merged box using weighted average
            merged_box = np.average(boxes, weights=weights, axis=0)
            
            # Calculate final score
            base_score = max(scores)  # Use highest confidence as base
            confidence_bonus = 0.1 * (len(current_group) - 1)  # Bonus for multiple detections
            final_score = min(base_score + view_bonus + confidence_bonus, 1.0)  # Cap at 1.0
            
            merged.append({
                'bbox': merged_box.tolist(),  # Convert to list for consistency
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
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            print(f"[DEBUG] Entering timeout wrapper for {func.__name__}")
            result = []
            error = []
            def target():
                try:
                    result.append(func(*args, **kwargs))
                except Exception as e:
                    error.append(e)
            thread = threading.Thread(target=target)
            thread.daemon = True
            thread.start()
            thread.join(seconds)
            if thread.is_alive():
                # Instead of stopping the thread directly, we'll just let it run in the background
                # since it's a daemon thread, it will be terminated when the main program exits
                raise TimeoutError(f"Operation timed out after {seconds} seconds")
            if error:
                raise error[0]
            return result[0]
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
    multi_view_params=None
):
    print("[DEBUG] Entered detect_crows_parallel")
    print(f"[DEBUG] detect_crows_parallel: {len(frames)} frames, score_threshold={score_threshold}, yolo_threshold={yolo_threshold}")
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
                            yolo_results = _run_model_inference(yolo_model, view.copy(), conf=yolo_threshold)[0]  # Make a copy of the view
                            print(f"[DEBUG] Frame {idx}: After YOLO model inference (multi-view)")
                            for bbox, score, cls in zip(yolo_results.boxes.xyxy, yolo_results.boxes.conf, yolo_results.boxes.cls):
                                print(f"[DEBUG] YOLO bbox: {bbox}, score: {score}, cls: {cls}")
                                if int(cls) == YOLO_BIRD_CLASS_ID:
                                    yolo_dets.append({
                                        'bbox': bbox.cpu().numpy(),
                                        'score': score.cpu().numpy(),
                                        'class': 'bird',
                                        'model': 'yolo',
                                        'view': 'multi'  # Mark as multi-view
                                    })
                        except TimeoutError:
                            logger.error("YOLO model inference timed out for multi-view")
                            continue  # Continue with next view instead of breaking
                else:
                    try:
                        print(f"[DEBUG] Frame {idx}: Before YOLO model inference (single-view)")
                        yolo_results = _run_model_inference(yolo_model, frame.copy(), conf=yolo_threshold)[0]  # Make a copy of the frame
                        print(f"[DEBUG] Frame {idx}: After YOLO model inference (single-view)")
                        print("[DEBUG] YOLO results:", yolo_results)
                        for bbox, score, cls in zip(yolo_results.boxes.xyxy, yolo_results.boxes.conf, yolo_results.boxes.cls):
                            print(f"[DEBUG] YOLO bbox: {bbox}, score: {score}, cls: {cls}")
                            if int(cls) == YOLO_BIRD_CLASS_ID:
                                yolo_dets.append({
                                    'bbox': bbox.cpu().numpy(),
                                    'score': score.cpu().numpy(),
                                    'class': 'bird',
                                    'model': 'yolo',
                                    'view': 'single'
                                })
                    except TimeoutError:
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
                                        'bbox': bbox.cpu().numpy(),
                                        'score': score.cpu().numpy(),
                                        'class': 'crow' if label == COCO_CROW_CLASS_ID else 'bird',
                                        'model': 'rcnn',
                                        'view': 'multi'  # Mark as multi-view
                                    })
                        except TimeoutError:
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
                                    'bbox': bbox.cpu().numpy(),
                                    'score': score.cpu().numpy(),
                                    'class': 'crow' if label == COCO_CROW_CLASS_ID else 'bird',
                                    'model': 'rcnn',
                                    'view': 'single'
                                })
                    except TimeoutError:
                        logger.error("RCNN model inference timed out")
                        rcnn_results = None
                print(f"[DEBUG] Frame {idx}: After RCNN detection")

            except Exception as e:
                print(f"[DEBUG] Frame {idx}: Exception occurred: {e}")
                logger.error(f"Error processing frame {idx}: {str(e)}", exc_info=True)
            finally:
                all_dets = yolo_dets + rcnn_dets
                if all_dets:
                    merged = merge_overlapping_detections(all_dets, iou_threshold=0.5)
                    filtered = [d for d in merged if d['score'] >= score_threshold and d['class'] in ('bird', 'crow')]
                    detections.append(filtered)
                else:
                    detections.append([])

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
    raise NotImplementedError("detect_crows is deprecated, use detect_crows_parallel instead.")
