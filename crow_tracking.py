import os
import json
import cv2
import numpy as np
from datetime import datetime, timedelta
import logging
from pathlib import Path
import torch
from tracking import load_faster_rcnn, load_triplet_model
from tracking import extract_crow_image
from collections import defaultdict
import shutil

logger = logging.getLogger(__name__)

class CrowTracker:
    def __init__(self, base_dir="crow_crops"):
        """Initialize the crow tracking system."""
        self.base_dir = Path(base_dir)
        self.crows_dir = self.base_dir / "crows"
        self.processing_dir = self.base_dir / "processing"
        self.metadata_dir = self.base_dir / "metadata"
        self.tracking_file = self.metadata_dir / "crow_tracking.json"
        
        # Create necessary directories
        self.crows_dir.mkdir(parents=True, exist_ok=True)
        self.processing_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize models
        logger.info("Loading detection model (Faster R-CNN)")
        self.detection_model = load_faster_rcnn()
        logger.info("Loading embedding model (Triplet Network)")
        self.embedding_model = load_triplet_model()
        
        # Load or create tracking data
        self.tracking_data = self._load_tracking_data()
        self.last_save_time = datetime.now()
        self.save_interval = 300  # Save every 5 minutes
        logger.info(f"Initialized CrowTracker with {len(self.tracking_data['crows'])} known crows")
        logger.info(f"Using base directory: {self.base_dir}")
    
    def _load_tracking_data(self):
        """Load tracking data from JSON file or create new if doesn't exist"""
        if self.tracking_file.exists():
            try:
                with open(self.tracking_file, 'r') as f:
                    data = json.load(f)
                logger.info(f"Loaded tracking data from {self.tracking_file}")
                return data
            except Exception as e:
                logger.error(f"Error loading tracking data: {e}")
        
        # Create new tracking data
        data = {
            "crows": {},  # crow_id -> {metadata}
            "last_id": 0,  # Last used crow ID
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }
        logger.info("Created new tracking data")
        return data
    
    def _save_tracking_data(self, force=False):
        """Save tracking data to JSON file"""
        try:
            current_time = datetime.now()
            # Save if forced or if enough time has passed
            if force or (current_time - self.last_save_time).total_seconds() >= self.save_interval:
                self.tracking_data["updated_at"] = current_time.isoformat()
                with open(self.tracking_file, 'w') as f:
                    json.dump(self.tracking_data, f, indent=2)
                self.last_save_time = current_time
                logger.info("Saved tracking data")
        except Exception as e:
            logger.error(f"Error saving tracking data: {e}")
    
    def _generate_crow_id(self):
        """Generate a new unique crow ID"""
        self.tracking_data["last_id"] += 1
        crow_id = f"crow_{self.tracking_data['last_id']:04d}"
        logger.info(f"Generated new crow ID: {crow_id}")
        return crow_id
    
    def find_matching_crow(self, crop):
        """Find a matching crow for the given crop using embedding similarity"""
        if not self.tracking_data["crows"]:
            logger.debug("No existing crows to match against")
            return None
        
        try:
            # Get embedding for new crop
            with torch.no_grad():
                new_embedding = self.embedding_model.get_embedding(crop)
                new_embedding = new_embedding.cpu().numpy().flatten()  # Ensure 1D array
            
            best_match = None
            best_score = -1
            threshold = 0.7  # Similarity threshold for matching
            
            # Compare with existing crows
            for crow_id, info in self.tracking_data["crows"].items():
                if "embedding" not in info:
                    logger.debug(f"Crow {crow_id} has no embedding, skipping")
                    continue
                
                # Load existing embedding and ensure it's 1D
                existing_embedding = np.array(info["embedding"]).flatten()
                
                # Calculate cosine similarity
                similarity = np.dot(new_embedding, existing_embedding) / (
                    np.linalg.norm(new_embedding) * np.linalg.norm(existing_embedding)
                )
                
                logger.debug(f"Similarity with crow {crow_id}: {similarity:.3f} (threshold: {threshold})")
                
                if similarity > best_score and similarity > threshold:
                    best_score = similarity
                    best_match = crow_id
            
            if best_match:
                logger.info(f"Found matching crow {best_match} with similarity {best_score:.3f}")
            else:
                logger.info(f"No matching crow found above threshold {threshold}. Best score was {best_score:.3f}")
            
            return best_match
            
        except Exception as e:
            logger.error(f"Error finding matching crow: {e}", exc_info=True)
            return None
    
    def process_detection(self, frame, frame_num, detection, video_path, frame_time):
        """Process a detection and either create a new crow or update an existing one."""
        try:
            # Convert frame_time from float to datetime if it's a float
            if isinstance(frame_time, float):
                # Assuming frame_time is seconds from video start
                frame_time = datetime.now() - timedelta(seconds=frame_time)
            
            # Extract crop
            crop = extract_crow_image(frame, detection['box'])
            if crop is None:
                logger.debug(f"Frame {frame_num}: Invalid crop")
                return None
            
            # Find matching crow
            crow_id = self.find_matching_crow(crop)
            
            # Create detection record
            detection_record = {
                "frame": frame_num,
                "box": detection['box'].tolist(),
                "score": float(detection['score']),
                "timestamp": frame_time.isoformat() if frame_time else None
            }
            
            if crow_id is None:
                # Create new crow
                crow_id = self._generate_crow_id()
                logger.info(f"Generated new crow ID: {crow_id}")
                
                self.tracking_data["crows"][crow_id] = {
                    "first_seen": frame_time.isoformat() if frame_time else None,
                    "last_seen": frame_time.isoformat() if frame_time else None,
                    "total_detections": 1,
                    "video_path": video_path,
                    "first_frame": frame_num,
                    "last_frame": frame_num,
                    "detections": [detection_record]  # Initialize with first detection
                }
                logger.info(f"Created new crow {crow_id}")
            else:
                # Update existing crow
                crow_info = self.tracking_data["crows"][crow_id]
                
                # Initialize detections list if it doesn't exist
                if "detections" not in crow_info:
                    crow_info["detections"] = []
                
                crow_info["last_seen"] = frame_time.isoformat() if frame_time else None
                crow_info["total_detections"] += 1
                crow_info["last_frame"] = frame_num
                crow_info["detections"].append(detection_record)
                logger.debug(f"Updated existing crow: {crow_id}")
            
            # Save crop
            crop_path = self.save_crop(crop, crow_id, frame_num)
            if crop_path:
                # Get and save embedding
                with torch.no_grad():
                    embedding = self.embedding_model.get_embedding(crop)
                    embedding = embedding.cpu().numpy().flatten()  # Ensure 1D array
                self.tracking_data["crows"][crow_id]["embedding"] = embedding.tolist()
                logger.debug(f"Saved embedding for crow {crow_id}")
            
            return crow_id
            
        except Exception as e:
            logger.error(f"Error processing detection: {str(e)}", exc_info=True)
            return None
    
    def get_crow_info(self, crow_id):
        """Get information about a specific crow"""
        return self.tracking_data["crows"].get(crow_id)
    
    def list_crows(self):
        """List all known crows with their metadata"""
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
        run_dir = self.processing_dir / f"run_{timestamp}"
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
            
            # Save the crop
            cv2.imwrite(str(crop_path), crop)
            logger.debug(f"Saved crop to {crop_path}")
            
            return crop_path
            
        except Exception as e:
            logger.error(f"Error saving crop: {e}", exc_info=True)
            return None 