#!/usr/bin/env python3
"""
Video orientation detection and correction.

Handles the common issue where mobile/camera videos have orientation metadata
that video players respect but OpenCV ignores.
"""

import cv2
import numpy as np
import subprocess
import json
import logging
from pathlib import Path
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

class VideoOrientationHandler:
    def __init__(self):
        self.orientation_cache = {}  # Cache orientations for performance
    
    def get_video_orientation(self, video_path: str) -> Tuple[int, bool]:
        """
        Determine the correct orientation for a video.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Tuple of (rotation_degrees, needs_flip)
            rotation_degrees: 0, 90, 180, or 270
            needs_flip: Whether to flip horizontally
        """
        video_path = str(video_path)
        
        # Check cache first
        if video_path in self.orientation_cache:
            return self.orientation_cache[video_path]
        
        rotation = 0
        needs_flip = False
        
        try:
            # Try to get orientation from metadata first
            rotation, needs_flip = self._get_orientation_from_metadata(video_path)
            
            # If no metadata found, use heuristics
            if rotation == 0 and not needs_flip:
                rotation, needs_flip = self._detect_orientation_heuristics(video_path)
                
        except Exception as e:
            logger.warning(f"Error detecting video orientation for {video_path}: {e}")
            rotation, needs_flip = 0, False
        
        # Cache the result
        self.orientation_cache[video_path] = (rotation, needs_flip)
        logger.info(f"Video orientation for {Path(video_path).name}: {rotation}° rotation, flip={needs_flip}")
        
        return rotation, needs_flip
    
    def _get_orientation_from_metadata(self, video_path: str) -> Tuple[int, bool]:
        """Extract orientation from video metadata using ffprobe."""
        try:
            cmd = [
                'ffprobe', 
                '-v', 'quiet',
                '-print_format', 'json',
                '-show_streams',
                '-show_format',
                video_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                return 0, False
                
            metadata = json.loads(result.stdout)
            
            # Look for rotation in stream metadata
            for stream in metadata.get('streams', []):
                if stream.get('codec_type') == 'video':
                    # Check tags for rotation
                    tags = stream.get('tags', {})
                    
                    # Common rotation fields
                    for field in ['rotate', 'rotation', 'Rotation']:
                        if field in tags:
                            try:
                                rotation = int(tags[field])
                                logger.info(f"Found rotation in metadata: {rotation}°")
                                return rotation, False
                            except (ValueError, TypeError):
                                continue
                    
                    # Check side data for rotation matrix
                    side_data = stream.get('side_data_list', [])
                    for data in side_data:
                        if 'displaymatrix' in data.get('side_data_type', '').lower():
                            # Parse display matrix if available
                            rotation = self._parse_display_matrix(data)
                            if rotation != 0:
                                logger.info(f"Found rotation in display matrix: {rotation}°")
                                return rotation, False
                    
                    break
                    
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, 
                FileNotFoundError, json.JSONDecodeError) as e:
            logger.debug(f"Could not read metadata with ffprobe: {e}")
        
        return 0, False
    
    def _parse_display_matrix(self, matrix_data: dict) -> int:
        """Parse display matrix to extract rotation."""
        # This is a simplified parser - real display matrices are more complex
        try:
            if 'rotation' in matrix_data:
                return int(matrix_data['rotation'])
        except (ValueError, TypeError):
            pass
        return 0
    
    def _detect_orientation_heuristics(self, video_path: str) -> Tuple[int, bool]:
        """
        Detect orientation using heuristics based on video properties.
        
        This handles the common case where mobile videos are recorded in portrait
        but don't have explicit rotation metadata.
        """
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return 0, False
            
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            
            # Detect portrait videos that should be rotated
            if height > width:
                # Portrait video - this is likely the issue!
                aspect_ratio = height / width
                
                # If it's a very tall video (common phone aspect ratios)
                if aspect_ratio > 1.3:  # More than 4:3 aspect ratio
                    logger.info(f"Detected portrait video ({width}x{height}) - applying 90° rotation")
                    return 90, False  # Rotate 90° clockwise to make it landscape
            
            # Check for very wide videos that might need 270° rotation
            elif width > height:
                aspect_ratio = width / height
                if aspect_ratio > 2.0:  # Very wide
                    # Could be a rotated portrait video
                    # This would need more sophisticated detection
                    pass
            
        except Exception as e:
            logger.warning(f"Error in orientation heuristics: {e}")
        
        return 0, False
    
    def apply_orientation_to_frame(self, frame: np.ndarray, rotation: int, flip: bool) -> np.ndarray:
        """
        Apply orientation correction to a frame.
        
        Args:
            frame: Input frame
            rotation: Rotation in degrees (0, 90, 180, 270)
            flip: Whether to flip horizontally
            
        Returns:
            Corrected frame
        """
        if frame is None:
            return None
        
        corrected = frame.copy()
        
        # Apply rotation
        if rotation == 90:
            corrected = cv2.rotate(corrected, cv2.ROTATE_90_CLOCKWISE)
        elif rotation == 180:
            corrected = cv2.rotate(corrected, cv2.ROTATE_180)
        elif rotation == 270:
            corrected = cv2.rotate(corrected, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        # Apply flip
        if flip:
            corrected = cv2.flip(corrected, 1)  # Horizontal flip
        
        return corrected
    
    def should_auto_correct_video(self, video_path: str) -> bool:
        """
        Determine if a video should be auto-corrected based on its properties.
        
        Returns True if the video is likely to benefit from orientation correction.
        """
        try:
            rotation, flip = self.get_video_orientation(video_path)
            return rotation != 0 or flip
        except Exception:
            return False

# Global instance for convenience
video_orientation_handler = VideoOrientationHandler()

def get_video_orientation(video_path: str) -> Tuple[int, bool]:
    """Convenience function to get video orientation."""
    return video_orientation_handler.get_video_orientation(video_path)

def apply_video_orientation(frame: np.ndarray, video_path: str) -> np.ndarray:
    """Convenience function to apply video orientation correction."""
    rotation, flip = video_orientation_handler.get_video_orientation(video_path)
    return video_orientation_handler.apply_orientation_to_frame(frame, rotation, flip)

def should_auto_correct_video(video_path: str) -> bool:
    """Convenience function to check if video should be auto-corrected."""
    return video_orientation_handler.should_auto_correct_video(video_path) 