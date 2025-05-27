# Crop Architecture Refactor: Video/Frame-Based Organization

## Overview

This document describes the major refactor of the crop saving architecture in the Facebeak crow identification system. The new architecture prevents training bias by organizing crops by video and frame instead of crow ID.

## Problem with Old Architecture

### Old Directory Structure (Biased)
```
crow_crops/
├── crows/
│   ├── crow_0001/
│   │   ├── crop_00000001_timestamp_videoname_frame_000123.jpg
│   │   ├── crop_00000002_timestamp_videoname_frame_000456.jpg
│   │   └── crop_00000003_timestamp_videoname_frame_000789.jpg
│   ├── crow_0002/
│   │   ├── crop_00000004_timestamp_videoname_frame_000234.jpg
│   │   └── crop_00000005_timestamp_videoname_frame_000567.jpg
│   └── crow_0003/
│       └── crop_00000006_timestamp_videoname_frame_000890.jpg
```

### Problems
1. **Training Bias**: Crops grouped by crow ID → model learns crow-specific features
2. **Temporal Loss**: Temporal relationships lost → model can't learn motion patterns
3. **Spatial Loss**: Spatial context lost → model can't learn environmental cues
4. **Unbalanced Data**: Some crows over-represented in training data
5. **Identity Overfitting**: Model learns to identify specific individuals rather than general crow features

## New Architecture (Bias-Free)

### New Directory Structure
```
crow_crops/
├── videos/
│   ├── video1/
│   │   ├── frame_000123_crop_001.jpg
│   │   ├── frame_000456_crop_001.jpg
│   │   ├── frame_000456_crop_002.jpg  # Multiple crows in same frame
│   │   └── frame_000789_crop_001.jpg
│   ├── video2/
│   │   ├── frame_000234_crop_001.jpg
│   │   └── frame_000567_crop_001.jpg
│   └── video3/
│       ├── frame_000100_crop_001.jpg
│       └── frame_000890_crop_001.jpg
├── metadata/
│   ├── crow_tracking.json      # Crow identity tracking (unchanged)
│   └── crop_metadata.json     # Maps crops to crow IDs
└── crows/                      # Legacy directory (backward compatibility)
```

### Benefits
1. **No Training Bias**: Crops organized by video/frame → prevents identity-specific overfitting
2. **Temporal Context**: Preserves temporal relationships → better motion pattern learning
3. **Spatial Context**: Maintains spatial relationships → better environmental learning
4. **Balanced Sampling**: Equal representation across videos → more robust training
5. **Metadata Tracking**: Still allows identity analysis when needed

## Implementation Details

### Core Changes

#### 1. CrowTracker Initialization
```python
# NEW: Videos directory for frame-based crop organization (prevents bias)
self.videos_dir = self.base_dir / "videos"
self.videos_dir.mkdir(exist_ok=True)

# NEW: Crop metadata file for mapping crops to crow IDs
self.crop_metadata_file = self.metadata_dir / "crop_metadata.json"
```

#### 2. New save_crop Method
```python
def save_crop(self, crop, crow_id, frame_num, video_path=None):
    """Save a crop image to disk using video/frame-based organization to prevent training bias."""
    
    # Extract video name
    video_name = Path(video_path).stem if video_path else "unknown"
    
    # Create video-specific directory (prevents bias by not grouping by crow ID)
    video_dir = self.videos_dir / video_name
    video_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate frame-based filename to prevent training bias
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
    
    # Save crop image
    cv2.imwrite(str(crop_path), crop_np)
    
    # Record crop metadata for tracking purposes (maintains crow ID mapping)
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
```

#### 3. Crop Metadata Structure
```json
{
  "crops": {
    "videos/video1/frame_000123_crop_001.jpg": {
      "crow_id": "crow_0001",
      "frame": 123,
      "video": "video1",
      "video_path": "/path/to/video1.mp4",
      "timestamp": "2025-05-27T00:47:55.721377"
    }
  },
  "created_at": "2025-05-27T00:47:55.721377",
  "updated_at": "2025-05-27T00:47:55.721377"
}
```

### Backward Compatibility

#### Helper Methods
```python
def get_crops_by_crow_id(self, crow_id):
    """Get all crop paths for a specific crow ID (for backward compatibility)."""
    
def get_crops_by_video(self, video_name):
    """Get all crop paths for a specific video."""
    
def get_crop_metadata_by_path(self, crop_path):
    """Get metadata for a specific crop path."""
```

#### Legacy Support
- Old `crows/` directory maintained for backward compatibility
- Existing tracking data format unchanged
- Migration utility provided for existing crops

## Training Benefits

### Machine Learning Advantages

1. **Reduced Overfitting**: Model learns general crow features, not individual identities
2. **Better Generalization**: Trained on diverse temporal and spatial contexts
3. **Balanced Datasets**: Equal sampling across videos prevents bias
4. **Temporal Learning**: Model can learn motion patterns and behaviors
5. **Environmental Context**: Model learns environmental cues and contexts

### Training Data Organization

#### Old Approach (Biased)
```python
# Training data grouped by crow ID - CAUSES BIAS
for crow_id in crow_dirs:
    crops = load_crops_from_crow_dir(crow_id)
    # Model learns crow-specific features!
```

#### New Approach (Unbiased)
```python
# Training data organized by video/frame - PREVENTS BIAS
for video in video_dirs:
    for frame_crops in load_frame_crops(video):
        # Model learns general crow features across contexts!
```

## Migration Guide

### For Existing Deployments

1. **Run Demo Script**:
   ```bash
   python utilities/crop_architecture_demo.py
   ```

2. **Analyze Legacy Crops**:
   ```bash
   python utilities/crop_architecture_demo.py --migrate
   ```

3. **Perform Migration** (when ready):
   ```bash
   python utilities/crop_architecture_demo.py --migrate --no-dry-run
   ```

### For New Deployments

- New architecture is automatically used
- No migration needed
- Better training data from day one

## Testing

### Verification Steps

1. **Architecture Demo**: Run `utilities/crop_architecture_demo.py`
2. **Directory Structure**: Verify `videos/` directory creation
3. **Metadata Files**: Check `crop_metadata.json` creation
4. **Crop Saving**: Test with GUI or command-line tools
5. **Backward Compatibility**: Verify legacy methods still work

### Expected Results

- Crops saved in `videos/{video_name}/frame_{frame:06d}_crop_{num:03d}.jpg`
- Metadata mapping maintained in `crop_metadata.json`
- Legacy `crows/` directory preserved
- Training bias eliminated

## Future Enhancements

### Planned Improvements

1. **Automatic Migration**: Complete legacy crop migration tool
2. **Training Integration**: Update training scripts for new architecture
3. **Analysis Tools**: Video-based analysis and visualization tools
4. **Performance Optimization**: Efficient metadata indexing
5. **Validation Tools**: Bias detection and prevention validation

### Research Opportunities

1. **Temporal Learning**: Leverage frame sequences for better models
2. **Spatial Context**: Use environmental cues for improved detection
3. **Balanced Sampling**: Develop optimal sampling strategies
4. **Cross-Video Learning**: Transfer learning across different environments

## Conclusion

The new video/frame-based crop architecture represents a significant improvement in the Facebeak system's approach to training data organization. By preventing training bias and preserving temporal/spatial context, this refactor enables the development of more robust and generalizable crow identification models.

The architecture maintains full backward compatibility while providing a clear path forward for bias-free machine learning training. This change will lead to better model performance and more reliable crow identification in diverse real-world scenarios. 