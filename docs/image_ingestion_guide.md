# Crow Image Ingestion Tool

The Crow Image Ingestion Tool allows you to process individual images, detect crows, extract 512x512 crops, and optionally label them during the ingestion process. This is useful for adding new training data to your crow detection system.

## Features

- **Image Processing**: Process individual images or entire directories
- **Crow Detection**: Uses the same detection pipeline as video processing
- **Crop Extraction**: Automatically extracts 512x512 crops around detected crows
- **Interactive Review**: Review each detection and approve/reject individual crops
- **Optional Labeling**: Add custom labels to crops during the review process
- **Recursive Search**: Option to search subdirectories for images
- **Progress Tracking**: Real-time progress updates and statistics

## Usage

### Running the Tool

#### Option 1: From Main Launcher (Recommended)
1. Run the main facebeak launcher:
   ```bash
   python facebeak.py
   ```
2. Navigate to the "Image Review" section
3. Click "Launch Image Ingestion" button

#### Option 2: Direct Launch
```bash
python run_image_ingestion.py
```

#### Option 3: Direct Script
```bash
python gui/image_ingestion_gui.py
```

### Step-by-Step Process

1. **Select Image Directory**: Choose the directory containing your images
2. **Select Output Directory**: Choose where to save the extracted crops
3. **Configure Settings**: Adjust detection parameters as needed
4. **Start Processing**: Click "Start Processing" to begin
5. **Review Detections**: For each image with detections:
   - View the image with detection boxes highlighted
   - Navigate between multiple detections if present
   - Add optional labels for each detection
   - Approve, reject, or skip detections
6. **Monitor Progress**: Watch real-time statistics and progress

### Detection Settings

- **Min Confidence**: Minimum confidence threshold for detections (0.1-0.9)
- **Multi-View Detection**: Enable multi-view detection for YOLO and/or Faster R-CNN
- **Orientation Correction**: Automatically correct crow orientation in crops
- **Recursive Search**: Search subdirectories for images
- **Box Merge Threshold**: Threshold for merging overlapping detections
- **BBox Padding**: Amount of padding around detected crows (0.1-0.8)

### Supported Image Formats

- JPEG (.jpg, .jpeg)
- PNG (.png)
- BMP (.bmp)
- TIFF (.tiff, .tif)

## Review Interface

When the tool finds detections in an image, it opens a review window where you can:

- **View the Image**: See the original image with detection boxes highlighted
- **Navigate Detections**: Use Previous/Next buttons to review multiple detections
- **Add Labels**: Optionally add custom labels for each detection
- **Approve & Save**: Save the crop with the current label
- **Reject**: Skip this detection without saving
- **Skip Image**: Skip the entire image

### Manual Bounding Box Drawing

The review interface now includes a powerful manual drawing feature:

- **Enable Manual Drawing**: Check the "Enable Manual Drawing" checkbox to activate drawing mode
- **Draw Square Bounding Boxes**: Click and drag on the image to draw square bounding boxes (rectangles are not allowed)
- **Visual Feedback**: 
  - Green boxes = Automatic detections
  - Red boxes = Manual detections  
  - Yellow highlight = Currently selected detection
- **Clear Manual Boxes**: Use the "Clear Manual Boxes" button to remove all manually drawn detections
- **Coordinate Conversion**: The tool automatically converts canvas coordinates to original image coordinates

#### How to Use Manual Drawing:

1. **Enable Drawing Mode**: Check the "Enable Manual Drawing" checkbox
2. **Draw Square Boxes**: Click and drag on the image to create square bounding boxes
3. **Review All Detections**: Navigate between automatic and manual detections using Previous/Next
4. **Label and Save**: Select a label from the dropdown and approve/reject detections
5. **Clear if Needed**: Use "Clear Manual Boxes" to start over

This feature is especially useful when:
- Automatic detection misses crows
- You want to add specific crops for training
- You need to correct detection boundaries
- You want to add multiple crops from a single image

### Labeling

Labels are selected from a predefined dropdown menu to ensure consistency:

#### Available Labels:
- **crow** - Standard crow detection (default)
- **not-crow** - False positive or non-crow object
- **juvenile-crow** - Young crow
- **adult-crow** - Mature crow
- **flying-crow** - Crow in flight
- **perching-crow** - Crow perched on surface
- **feeding-crow** - Crow feeding
- **multiple-crows** - Multiple crows in single detection
- **crow-partial** - Partially visible crow
- **crow-occluded** - Crow partially hidden by objects
- **crow-blurry** - Blurry or low-quality crow image
- **crow-high-quality** - High-quality, clear crow image

Labels are stored in the tracking metadata and can be used to:
- Categorize different types of crows
- Mark specific behaviors or poses
- Organize data for different training purposes
- Filter training data by quality or characteristics

## Output Structure

The tool saves crops in the same structure as the video processing tool:

```
output_directory/
├── crows/
│   ├── video_name/
│   │   ├── frame_000001/
│   │   │   ├── crop_000001.jpg
│   │   │   └── crop_000002.jpg
│   │   └── frame_000002/
│   │       └── crop_000001.jpg
│   └── ...
├── metadata/
│   ├── crow_tracking.json
│   └── crop_metadata.json
└── ...
```

## Statistics

The tool tracks several statistics:

- **Images Processed**: Total number of images processed
- **Detections Found**: Total number of crow detections found
- **Crops Saved**: Number of crops successfully saved
- **Crops Rejected**: Number of detections rejected by user
- **Images Skipped**: Number of images skipped (no detections or errors)

## Tips for Best Results

1. **Image Quality**: Use high-quality images for better detection accuracy
2. **Crow Visibility**: Ensure crows are clearly visible and not heavily occluded
3. **Lighting**: Good lighting conditions improve detection performance
4. **Labeling**: Use consistent labels for similar types of crows
5. **Batch Processing**: Process multiple images at once for efficiency

## Troubleshooting

### Common Issues

- **No Detections Found**: Try lowering the confidence threshold
- **Poor Detection Quality**: Check image quality and lighting
- **Slow Processing**: Reduce batch size or disable multi-view detection
- **Memory Issues**: Process fewer images at once

### Error Messages

- **"Could not load image"**: Check file format and file integrity
- **"No image files found"**: Verify directory path and file extensions
- **"Failed to extract crop"**: Detection box may be invalid or too small

## Integration with Training Pipeline

The extracted crops and labels can be used directly with your training pipeline:

1. **Crops**: The 512x512 crops are ready for training
2. **Labels**: Custom labels can be used for supervised learning
3. **Metadata**: Tracking data includes detection confidence and metadata
4. **Organization**: Video/frame-based organization prevents training bias

## Advanced Usage

### Custom Labels

You can use labels to organize your data for different training purposes:

- **Behavior Labels**: "perching", "flying", "feeding"
- **Pose Labels**: "side_view", "front_view", "back_view"
- **Quality Labels**: "high_quality", "blurry", "occluded"
- **Individual Labels**: "crow_001", "crow_002", etc.

### Batch Processing

For large datasets:

1. Organize images into subdirectories
2. Enable recursive search
3. Process in batches to manage memory usage
4. Use consistent labeling conventions

### Integration with Existing Data

The tool integrates seamlessly with existing crow tracking data:

- New crops are added to the existing database
- Crow IDs are assigned consistently
- Metadata is preserved and updated
- No conflicts with existing training data 