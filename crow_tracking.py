import os
import json
import cv2
import numpy as np
from datetime import datetime, timedelta
import logging
from pathlib import Path
import torch
from tracking import load_faster_rcnn, load_triplet_model
from tracking import extract_normalized_crow_crop
from collections import defaultdict
import shutil

logger = logging.getLogger(__name__)

class CrowTracker:
    def __init__(self, base_dir="crow_crops", similarity_threshold=0.7):
        """Initialize the crow tracker.
        
        Args:
            base_dir: Base directory for storing crow data
            similarity_threshold: Cosine similarity threshold for matching crows (default: 0.7)
        """
        self.base_dir = Path(base_dir)
        self.similarity_threshold = similarity_threshold
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize directories
        self.crows_dir = self.base_dir / "crows"
        self.crows_dir.mkdir(exist_ok=True)
        self.processing_dir = self.base_dir / "processing"  # Changed from processing_runs
        self.processing_dir.mkdir(exist_ok=True)
        self.metadata_dir = self.base_dir / "metadata"
        self.metadata_dir.mkdir(exist_ok=True)
        
        # Initialize tracking file path
        self.tracking_file = self.metadata_dir / "crow_tracking.json"
        
        # Load detection model
        logger.info("Loading detection model (Faster R-CNN)")
        self.detection_model = load_faster_rcnn()
        
        # Load embedding model
        logger.info("Loading embedding model (Triplet Network)")
        self.embedding_model = load_triplet_model()
        
        # Load or create tracking data
        self.tracking_data = self._load_tracking_data()
        
        # Log initialization
        logger.info(f"Initialized CrowTracker with {len(self.tracking_data['crows'])} known crows")
        logger.info(f"Using base directory: {self.base_dir}")
    
    def _load_tracking_data(self):
        """Load tracking data from file or create new if not exists."""
        try:
            if self.tracking_file.exists():
                with open(self.tracking_file, 'r') as f:
                    data = json.load(f)
                    # Ensure last_id is at root level
                    if "last_id" in data.get("metadata", {}):
                        data["last_id"] = data["metadata"]["last_id"]
                        del data["metadata"]["last_id"]
                    # Ensure last_id exists
                    if "last_id" not in data:
                        data["last_id"] = 0
                    logger.info(f"Loaded tracking data from {self.tracking_file}")
                    return data
            else:
                # Create new tracking data
                data = {
                    "crows": {},
                    "last_id": 0,
                    "created_at": datetime.now().isoformat(),
                    "updated_at": datetime.now().isoformat()
                }
                logger.info("Created new tracking data")
                # Save initial tracking data
                self._save_tracking_data(data, force=True)
                logger.info(f"Saved initial tracking data to {self.tracking_file}")
                return data
        except Exception as e:
            logger.error(f"Error loading tracking data: {str(e)}")
            # Create new tracking data on error
            data = {
                "crows": {},
                "last_id": 0,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat()
            }
            logger.info("Created new tracking data after error")
            return data
    
    def _save_tracking_data(self, data=None, force=False):
        """Save tracking data to file.
        
        Args:
            data: Optional data to save. If None, saves current tracking_data
            force: Whether to force save even if no changes
        """
        try:
            if data is None:
                data = self.tracking_data
            
            # Update timestamp
            data["updated_at"] = datetime.now().isoformat()
            
            # Ensure directory exists
            self.tracking_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Save to file
            with open(self.tracking_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info("Tracking data saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving tracking data: {str(e)}")
            raise
    
    def _generate_crow_id(self):  # Changed from generate_crow_id to match test
        """Generate a new unique crow ID."""
        try:
            # Increment last_id
            self.tracking_data["last_id"] += 1
            crow_id = f"crow_{self.tracking_data['last_id']:04d}"
            
            # Save tracking data
            self._save_tracking_data()
            
            return crow_id
            
        except Exception as e:
            logger.error(f"Error generating crow ID: {str(e)}")
            return None
    
    def find_matching_crow(self, crop):
        """Find a matching crow based on embedding similarity.
        
        Args:
            crop: Dictionary containing 'full' and 'head' tensors or numpy arrays
            
        Returns:
            str: Crow ID if match found, None otherwise
        """
        try:
            if crop is None or 'full' not in crop:
                return None
                
            # Get embedding for the new crop
            with torch.no_grad():
                # Handle both numpy array and tensor formats
                crop_data = crop['full']
                if isinstance(crop_data, np.ndarray):
                    # Convert numpy array [H, W, C] to tensor [C, H, W]
                    crop_tensor = torch.from_numpy(crop_data).float()
                    crop_tensor = crop_tensor.permute(2, 0, 1)  # [H, W, C] -> [C, H, W]
                else:
                    # Already a tensor, ensure correct format
                    crop_tensor = crop_data
                    if len(crop_tensor.shape) == 4:  # Remove batch dimension [1, C, H, W] -> [C, H, W]
                        crop_tensor = crop_tensor.squeeze(0)
                
                new_embedding = self.embedding_model.get_embedding(crop_tensor)
                new_embedding = new_embedding.cpu().numpy().flatten()
                new_embedding = new_embedding / np.linalg.norm(new_embedding)
            
            # Compare with existing crows
            best_match = None
            best_similarity = self.similarity_threshold  # Use instance threshold
            
            for crow_id, crow_data in self.tracking_data["crows"].items():
                if "embedding" not in crow_data or crow_data["embedding"] is None:
                    continue
                    
                # Get existing embedding
                existing_embedding = np.array(crow_data["embedding"])
                existing_embedding = existing_embedding / np.linalg.norm(existing_embedding)
                
                # Calculate cosine similarity
                similarity = np.dot(new_embedding, existing_embedding)
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = crow_id
            
            return best_match
            
        except Exception as e:
            logger.error(f"Error finding matching crow: {str(e)}")
            return None
    
    def process_detection(self, frame, frame_num, detection, video_path, frame_time):
        """Process a detection and either create a new crow or update an existing one."""
        try:
            # Convert detection to proper format - HANDLE BOTH DICT AND ARRAY
            if isinstance(detection, dict):
                box = np.array(detection['bbox'], dtype=np.float32)
                score = detection['score']
            else:
                # Handle numpy array format [[x1, y1, x2, y2, score]]
                if isinstance(detection, np.ndarray):
                    if len(detection.shape) == 2:  # 2D array
                        detection = detection[0]  # Get first row
                    box = detection[:4].astype(np.float32)
                    score = float(detection[4])
                else:
                    raise ValueError(f"Unsupported detection format: {type(detection)}")
            
            # Validate box coordinates
            if not isinstance(box, (list, tuple, np.ndarray)) or len(box) != 4:
                logger.warning(f"Invalid box format: {box}")
                return None
            
            # Validate detection score
            if not (0.0 <= score <= 1.0):
                logger.warning(f"Invalid detection score: {score}")
                return None
            
            # Check if box is within frame bounds
            h, w = frame.shape[:2]
            if (box[0] >= box[2] or box[1] >= box[3] or  # Invalid box dimensions
                box[0] < 0 or box[1] < 0 or  # Box outside frame (left/top)
                box[2] > w or box[3] > h or  # Box outside frame (right/bottom)
                box[2] - box[0] < 10 or box[3] - box[1] < 10):  # Box too small
                logger.warning(f"Invalid box coordinates: {box} for frame size {w}x{h}")
                return None
            
            # Convert frame_time from float to datetime if it's a float
            if isinstance(frame_time, float):
                # Assuming frame_time is seconds from video start
                frame_time = datetime.now() - timedelta(seconds=frame_time)
            
            # Extract crop
            crop = extract_normalized_crow_crop(frame, box)
            if crop is None:
                logger.debug(f"Frame {frame_num}: Failed to extract crop")
                return None
            
            # Find matching crow
            crow_id = self.find_matching_crow(crop)
            
            # Create detection record
            detection_record = {
                "frame": frame_num,
                "bbox": box.tolist(),
                "score": float(score),
                "timestamp": frame_time.isoformat() if frame_time else None
            }
            
            if crow_id is None:
                # Create new crow
                crow_id = self._generate_crow_id()  # Changed to match test
                if crow_id is None:
                    logger.error("Failed to generate crow ID")
                    return None
                
                # Initialize crow data
                self.tracking_data["crows"][crow_id] = {
                    "detections": [detection_record],
                    "total_detections": 1,
                    "first_frame": frame_num,  # Added to match test
                    "last_frame": frame_num,   # Added to match test
                    "first_seen": frame_time.isoformat() if frame_time else None,
                    "last_seen": frame_time.isoformat() if frame_time else None,
                    "video_path": str(video_path) if video_path else None,
                    "embedding": None  # Will be set after saving crop
                }
                
                # Save crop and get embedding
                crop_path = self.save_crop(crop, crow_id, frame_num)
                if crop_path:
                    # Get and save embedding
                    with torch.no_grad():
                        # Handle both numpy array and tensor formats
                        crop_data = crop['full']
                        if isinstance(crop_data, np.ndarray):
                            # Convert numpy array [H, W, C] to tensor [C, H, W]
                            crop_tensor = torch.from_numpy(crop_data).float()
                            crop_tensor = crop_tensor.permute(2, 0, 1)  # [H, W, C] -> [C, H, W]
                        else:
                            # Already a tensor, ensure correct format
                            crop_tensor = crop_data
                            if len(crop_tensor.shape) == 4:  # Remove batch dimension [1, C, H, W] -> [C, H, W]
                                crop_tensor = crop_tensor.squeeze(0)
                        
                        embedding = self.embedding_model.get_embedding(crop_tensor)
                        embedding = embedding.cpu().numpy().flatten()  # Ensure 1D array
                        embedding = embedding / np.linalg.norm(embedding)  # Normalize
                    self.tracking_data["crows"][crow_id]["embedding"] = embedding.tolist()
                    logger.debug(f"Saved embedding for crow {crow_id}")
            else:
                # Update existing crow
                crow_data = self.tracking_data["crows"][crow_id]
                crow_data["detections"].append(detection_record)
                crow_data["total_detections"] += 1
                crow_data["last_frame"] = frame_num  # Added to match test
                crow_data["last_seen"] = frame_time.isoformat() if frame_time else None
                
                # Save crop periodically (every 10 detections)
                if crow_data["total_detections"] % 10 == 0:
                    crop_path = self.save_crop(crop, crow_id, frame_num)
                    if crop_path:
                        # Update embedding
                        with torch.no_grad():
                            # Handle both numpy array and tensor formats
                            crop_data = crop['full']
                            if isinstance(crop_data, np.ndarray):
                                # Convert numpy array [H, W, C] to tensor [C, H, W]
                                crop_tensor = torch.from_numpy(crop_data).float()
                                crop_tensor = crop_tensor.permute(2, 0, 1)  # [H, W, C] -> [C, H, W]
                            else:
                                # Already a tensor, ensure correct format
                                crop_tensor = crop_data
                                if len(crop_tensor.shape) == 4:  # Remove batch dimension [1, C, H, W] -> [C, H, W]
                                    crop_tensor = crop_tensor.squeeze(0)
                            
                            embedding = self.embedding_model.get_embedding(crop_tensor)
                            embedding = embedding.cpu().numpy().flatten()  # Ensure 1D array
                            embedding = embedding / np.linalg.norm(embedding)  # Normalize
                        crow_data["embedding"] = embedding.tolist()
                        logger.debug(f"Updated embedding for crow {crow_id}")
            
            # Save tracking data periodically
            self._save_tracking_data()
            
            return crow_id
            
        except Exception as e:
            logger.error(f"Error processing detection: {str(e)}", exc_info=True)
            return None
    
    def get_crow_info(self, crow_id):
        """Get information about a specific crow."""
        return self.tracking_data["crows"].get(crow_id)
    
    def list_crows(self):
        """List all known crows with their metadata."""
        return {
            crow_id: {
                "total_detections": data["total_detections"],
                "first_seen": data["first_seen"],
                "last_seen": data["last_seen"],
                "video_path": data.get("video_path", None)  # Use get() to handle missing key
            }
            for crow_id, data in self.tracking_data["crows"].items()
        }
    
    def create_processing_run(self):
        """Create a new processing run directory with timestamp."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = self.processing_dir / f"run_{timestamp}"  # Changed from processing_runs
        run_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created processing run directory: {run_dir}")
        return run_dir
    
    def cleanup_processing_dir(self, run_dir):
        """Clean up a processing run directory."""
        try:
            if run_dir.exists():
                shutil.rmtree(run_dir)
                logger.info(f"Cleaned up processing directory: {run_dir}")
        except Exception as e:
            logger.error(f"Error cleaning up processing directory: {e}")
    
    def save_crop(self, crop, crow_id, frame_num):
        """Save a crop image to disk and return the path."""
        try:
            # Create crow directory if it doesn't exist
            crow_dir = self.crows_dir / crow_id
            crow_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate filename with frame number
            filename = f"frame_{frame_num:06d}.jpg"
            crop_path = crow_dir / filename
            
            # Handle both numpy array and tensor formats
            if isinstance(crop, dict):
                crop_data = crop['full']
                if isinstance(crop_data, np.ndarray):
                    # Numpy array format [H, W, C] normalized [0,1] -> [0,255] uint8
                    crop_np = (crop_data * 255).astype(np.uint8)
                else:
                    # Tensor format - convert to numpy
                    crop_tensor = crop_data
                    if len(crop_tensor.shape) == 4:  # Remove batch dimension [1, C, H, W] -> [C, H, W]
                        crop_tensor = crop_tensor.squeeze(0)
                    # Convert from [C, H, W] to [H, W, C] and scale to 0-255
                    crop_np = (crop_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            else:
                # Direct numpy array
                crop_np = crop
            
            # Save the crop (OpenCV expects BGR format, but for saving it should be fine)
            cv2.imwrite(str(crop_path), crop_np)
            logger.debug(f"Saved crop to {crop_path}")
            
            return crop_path
            
        except Exception as e:
            logger.error(f"Error saving crop: {e}", exc_info=True)
            return None 