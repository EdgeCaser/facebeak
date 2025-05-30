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
from audio import extract_and_save_crow_audio

logger = logging.getLogger(__name__)

class CrowTracker:
    def __init__(self, base_dir="crow_crops", similarity_threshold=0.7, enable_audio_extraction=True, audio_duration=2.0, correct_orientation=True):
        """
        Initialize CrowTracker with optional audio extraction and orientation correction.
        
        Args:
            base_dir: Base directory for storing crow data
            similarity_threshold: Threshold for matching crow embeddings
            enable_audio_extraction: Whether to extract audio segments during processing
            audio_duration: Duration of audio segments to extract (seconds)
            correct_orientation: Whether to auto-correct crow crop orientation
        """
        self.base_dir = Path(base_dir)
        self.similarity_threshold = similarity_threshold
        self.enable_audio_extraction = enable_audio_extraction
        self.audio_duration = audio_duration
        self.correct_orientation = correct_orientation
        
        # Create directory structure - NEW: video/frame-based instead of crow-based to prevent training bias
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # NEW: Videos directory for frame-based crop organization (prevents bias)
        self.videos_dir = self.base_dir / "videos"
        self.videos_dir.mkdir(exist_ok=True)
        
        # Keep legacy crows directory for backward compatibility with existing tracking
        self.crows_dir = self.base_dir / "crows"
        self.crows_dir.mkdir(exist_ok=True)
        
        self.processing_dir = self.base_dir / "processing"
        self.processing_dir.mkdir(exist_ok=True)
        self.metadata_dir = self.base_dir / "metadata"
        self.metadata_dir.mkdir(exist_ok=True)
        
        # Audio directory
        if self.enable_audio_extraction:
            self.audio_dir = self.base_dir / "audio"
            self.audio_dir.mkdir(exist_ok=True)
            logger.info(f"Audio extraction enabled. Audio directory: {self.audio_dir}")
        else:
            self.audio_dir = None
            logger.info("Audio extraction disabled")
        
        # Orientation correction
        logger.info(f"Orientation correction: {'enabled' if self.correct_orientation else 'disabled'}")
        
        # Tracking data file
        self.tracking_file = self.metadata_dir / "crow_tracking.json"
        
        # NEW: Crop metadata file for mapping crops to crow IDs
        self.crop_metadata_file = self.metadata_dir / "crop_metadata.json"
        
        # Load detection model
        logger.info("Loading detection model (Faster R-CNN)")
        self.detection_model = load_faster_rcnn()
        
        # Load embedding model
        logger.info("Loading embedding model (Triplet Network)")
        self.embedding_model = load_triplet_model()
        
        # Load or create tracking data
        self.tracking_data = self._load_tracking_data()
        
        # NEW: Load or create crop metadata
        self.crop_metadata = self._load_crop_metadata()
        
        # Log initialization
        logger.info(f"Initialized CrowTracker with {len(self.tracking_data['crows'])} known crows")
        logger.info(f"Using base directory: {self.base_dir}")
        logger.info(f"NEW: Using video/frame-based crop organization to prevent training bias")
    
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
                    # Ensure last_crop_id exists for backward compatibility
                    if "last_crop_id" not in data:
                        data["last_crop_id"] = 0
                        logger.info("Added last_crop_id for backward compatibility")
                    logger.info(f"Loaded tracking data from {self.tracking_file}")
                    return data
            else:
                # Create new tracking data
                data = {
                    "crows": {},
                    "last_id": 0,
                    "last_crop_id": 0,  # Global crop counter for uniqueness
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
                "last_crop_id": 0,  # Global crop counter for uniqueness
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
    
    def _load_crop_metadata(self):
        """Load crop metadata from file or create new if not exists."""
        try:
            if self.crop_metadata_file.exists():
                with open(self.crop_metadata_file, 'r') as f:
                    data = json.load(f)
                    logger.info(f"Loaded crop metadata from {self.crop_metadata_file}")
                    return data
            else:
                # Create new crop metadata
                data = {
                    "crops": {},  # crop_path -> {"crow_id": str, "frame": int, "video": str, "timestamp": str}
                    "created_at": datetime.now().isoformat(),
                    "updated_at": datetime.now().isoformat()
                }
                logger.info("Created new crop metadata")
                self._save_crop_metadata(data, force=True)
                return data
        except Exception as e:
            logger.error(f"Error loading crop metadata: {str(e)}")
            # Create new crop metadata on error
            data = {
                "crops": {},
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat()
            }
            logger.info("Created new crop metadata after error")
            return data
    
    def _save_crop_metadata(self, data=None, force=False):
        """Save crop metadata to file."""
        try:
            if data is None:
                data = self.crop_metadata
            
            # Update timestamp
            data["updated_at"] = datetime.now().isoformat()
            
            # Ensure directory exists
            self.crop_metadata_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Save to file
            with open(self.crop_metadata_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.debug("Crop metadata saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving crop metadata: {str(e)}")
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
            crop = extract_normalized_crow_crop(frame, box, correct_orientation=self.correct_orientation, padding=0.3)
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
                "timestamp": frame_time.isoformat() if frame_time else None,
                "video_path": str(video_path) if video_path else None,
                "crop_filename": None  # Will be set when crop is saved
            }
            
            # Get FPS for audio extraction
            fps = None
            if self.enable_audio_extraction and video_path:
                try:
                    cap = cv2.VideoCapture(str(video_path))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    cap.release()
                except Exception as e:
                    logger.warning(f"Could not get FPS from video: {e}")
                    fps = 30.0  # Default FPS
            
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
                crop_path = self.save_crop(crop, crow_id, frame_num, video_path)
                if crop_path:
                    # Record crop filename in detection record
                    detection_record["crop_filename"] = crop_path.name
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
                
                # Extract and save audio if enabled
                if self.enable_audio_extraction and video_path and fps:
                    try:
                        # Calculate frame time in seconds from video start
                        frame_time_seconds = frame_num / fps
                        audio_path = extract_and_save_crow_audio(
                            video_path, frame_time_seconds, fps, crow_id, frame_num, 
                            self.audio_dir, self.audio_duration
                        )
                        if audio_path:
                            logger.debug(f"Saved audio for new crow {crow_id}: {audio_path}")
                    except Exception as e:
                        logger.warning(f"Failed to extract audio for new crow {crow_id}: {e}")
                        
            else:
                # Update existing crow
                crow_data = self.tracking_data["crows"][crow_id]
                crow_data["detections"].append(detection_record)
                crow_data["total_detections"] += 1
                crow_data["last_frame"] = frame_num  # Added to match test
                crow_data["last_seen"] = frame_time.isoformat() if frame_time else None
                
                # Save crop periodically (every 10 detections)
                if crow_data["total_detections"] % 10 == 0:
                    crop_path = self.save_crop(crop, crow_id, frame_num, video_path)
                    if crop_path:
                        # Record crop filename in detection record
                        detection_record["crop_filename"] = crop_path.name
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
                    
                    # Extract and save audio if enabled
                    if self.enable_audio_extraction and video_path and fps:
                        try:
                            # Calculate frame time in seconds from video start
                            frame_time_seconds = frame_num / fps
                            audio_path = extract_and_save_crow_audio(
                                video_path, frame_time_seconds, fps, crow_id, frame_num, 
                                self.audio_dir, self.audio_duration
                            )
                            if audio_path:
                                logger.debug(f"Saved audio for existing crow {crow_id}: {audio_path}")
                        except Exception as e:
                            logger.warning(f"Failed to extract audio for existing crow {crow_id}: {e}")
            
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
    
    def get_crops_by_crow_id(self, crow_id):
        """Get all crop paths for a specific crow ID (for backward compatibility)."""
        crop_paths = []
        for crop_path, metadata in self.crop_metadata["crops"].items():
            if metadata["crow_id"] == crow_id:
                full_path = self.base_dir / crop_path
                if full_path.exists():
                    crop_paths.append(str(full_path))
        return sorted(crop_paths)
    
    def get_crops_by_video(self, video_name):
        """Get all crop paths for a specific video."""
        crop_paths = []
        for crop_path, metadata in self.crop_metadata["crops"].items():
            if metadata["video"] == video_name:
                full_path = self.base_dir / crop_path
                if full_path.exists():
                    crop_paths.append(str(full_path))
        return sorted(crop_paths)
    
    def get_crop_metadata_by_path(self, crop_path):
        """Get metadata for a specific crop path."""
        # Convert absolute path to relative if needed
        if Path(crop_path).is_absolute():
            try:
                crop_relative_path = str(Path(crop_path).relative_to(self.base_dir))
            except ValueError:
                return None
        else:
            crop_relative_path = crop_path
        
        return self.crop_metadata["crops"].get(crop_relative_path)
    
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
    
    def save_crop(self, crop, crow_id, frame_num, video_path=None):
        """Save a crop image to disk using video/frame-based organization to prevent training bias."""
        try:
            # Extract video name if provided
            video_name = "unknown"
            if video_path:
                video_name = Path(video_path).stem  # Get filename without extension
                # Sanitize video name for filesystem
                video_name = "".join(c for c in video_name if c.isalnum() or c in ('-', '_'))[:20]
            
            # NEW: Create video-specific directory (prevents bias by not grouping by crow ID)
            video_dir = self.videos_dir / video_name
            video_dir.mkdir(parents=True, exist_ok=True)
            
            # NEW: Generate frame-based filename to prevent training bias
            # Format: frame_XXXXXX_crop_XXX.jpg (multiple crops per frame possible)
            base_filename = f"frame_{frame_num:06d}_crop"
            
            # Find next available crop number for this frame
            crop_counter = 1
            while True:
                filename = f"{base_filename}_{crop_counter:03d}.jpg"
                crop_path = video_dir / filename
                if not crop_path.exists():
                    break
                crop_counter += 1
            
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
            
            # NEW: Record crop metadata for tracking purposes (maintains crow ID mapping)
            crop_relative_path = str(crop_path.relative_to(self.base_dir))
            self.crop_metadata["crops"][crop_relative_path] = {
                "crow_id": crow_id,
                "frame": frame_num,
                "video": video_name,
                "video_path": str(video_path) if video_path else None,
                "timestamp": datetime.now().isoformat()
            }
            
            # Save crop metadata
            self._save_crop_metadata()
            
            logger.debug(f"Saved crop to {crop_path} (video/frame-based organization)")
            logger.debug(f"Mapped crop to crow {crow_id} in metadata")
            
            return crop_path
            
        except Exception as e:
            logger.error(f"Error saving crop: {e}", exc_info=True)
            return None 