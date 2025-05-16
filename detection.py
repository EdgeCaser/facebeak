import torch
import torchvision
from tqdm import tqdm

# Load model once at module level
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
model.eval()
if torch.cuda.is_available():
    model = model.cuda()
    print("[INFO] Detection model loaded on GPU")
else:
    print("[INFO] Detection model loaded on CPU")

# COCO dataset class IDs
COCO_BIRD_CLASS_ID = 16  # General bird class
COCO_CROW_CLASS_ID = 20  # Specific crow class (if available in the model)

def detect_crows(frames, score_threshold=0.5):
    """
    Detect birds in video frames using Faster R-CNN.
    Args:
        frames: List of video frames
        score_threshold: Minimum confidence score for detections
    Returns:
        List of detections per frame
    """
    detections = []
    print(f"[INFO] Processing {len(frames)} frames for bird detection")
    
    with torch.no_grad():
        for frame in tqdm(frames, desc="Detecting birds"):
            # Convert frame to tensor and normalize
            frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
            if torch.cuda.is_available():
                frame_tensor = frame_tensor.cuda()
            
            # Get predictions
            pred = model([frame_tensor])
            
            # Filter for bird detections
            bird_detections = []
            for box, label, score in zip(pred[0]['boxes'], pred[0]['labels'], pred[0]['scores']):
                # Check for either general bird or specific crow class
                if (label == COCO_BIRD_CLASS_ID or label == COCO_CROW_CLASS_ID) and score > score_threshold:
                    bird_detections.append({
                        'box': box.cpu().numpy(),
                        'score': score.cpu().numpy(),
                        'class': 'crow' if label == COCO_CROW_CLASS_ID else 'bird'
                    })
            
            detections.append(bird_detections)
    
    total_detections = sum(len(d) for d in detections)
    frames_with_birds = sum(1 for d in detections if d)
    print(f"[INFO] Detection complete. Found {total_detections} birds in {frames_with_birds} frames")
    return detections
