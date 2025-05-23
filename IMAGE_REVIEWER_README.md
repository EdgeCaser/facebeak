# Crow Image Reviewer

The Crow Image Reviewer is a tool for manually reviewing and labeling crow images to improve training data quality. It allows users to quickly confirm whether images detected as crows by the model are actually crows or false positives.

## Features

- **Quick Review**: Load up to 100 unlabeled images at a time for efficient batch review
- **Keyboard Shortcuts**: Fast labeling with number keys (1=Crow, 2=Not a Crow, 3=Not Sure)
- **Training Data Management**: Images marked as "Not a crow" are automatically excluded from training data
- **Progress Tracking**: Visual progress bar and statistics showing review progress
- **Database Integration**: All labels are stored in the database for future reference

## How to Use

### Option 1: Launch from GUI Launcher
1. Run the main GUI launcher: `python gui_launcher.py`
2. Look for the "Image Review" section
3. Click "Launch Image Reviewer"

### Option 2: Run Directly
```bash
python image_reviewer.py
```

## Interface Overview

### Left Panel (Controls)
- **Image Directory**: Select the directory containing crow crop images (defaults to `crow_crops`)
- **Progress**: Shows current image number and progress bar
- **Navigation**: Previous/Next buttons to move between images
- **Labeling**: Radio buttons to select label type
- **Statistics**: Shows current labeling statistics

### Right Panel (Image Display)
- Large image display area
- Image filename and information
- Automatically scales images to fit display

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `1` | Mark as Crow and move to next |
| `2` | Mark as Not a Crow and move to next |
| `3` | Mark as Not Sure and move to next |
| `←` | Previous image |
| `→` | Next image |
| `Enter` | Submit current label |

## Label Types

### Crow
- Use for images that clearly contain crows
- These images will be included in training data
- Helps improve model accuracy for positive examples

### Not a Crow
- Use for false positives (birds that aren't crows, objects misidentified as crows, etc.)
- These images are automatically excluded from training data
- Helps reduce noise in the training dataset

### Not Sure
- Use when you're uncertain about the identification
- These images are flagged for potential manual review by experts
- Can be revisited later for final classification

## Database Storage

All labels are stored in the SQLite database with the following information:
- Image path
- Label (crow/not_a_crow/not_sure)
- Confidence score (if provided)
- Reviewer notes (optional)
- Timestamp
- Training data inclusion flag

## Workflow Recommendations

1. **Start with High-Confidence Images**: Begin with images that are clearly crows or clearly not crows
2. **Use Keyboard Shortcuts**: For fastest review, use number keys to label and advance
3. **Regular Breaks**: Take breaks every 50-100 images to maintain accuracy
4. **Review Statistics**: Check the statistics panel periodically to track progress
5. **Batch Processing**: Review images in batches of 20-100 for efficiency

## Integration with Training Pipeline

Images labeled as "not_a_crow" are automatically excluded from training data by setting `is_training_data=False` in the database. This ensures that:

1. **Cleaner Training Data**: False positives don't contaminate the training set
2. **Improved Model Performance**: The model learns from higher-quality examples
3. **Feedback Loop**: Poor detections are identified and removed from future training

## Troubleshooting

### No Images Found
- Check that the image directory path is correct
- Ensure the directory contains `.jpg`, `.jpeg`, or `.png` files
- Verify that images haven't already been labeled (the tool only shows unlabeled images)

### Database Errors
- Ensure the database is accessible and not corrupted
- Check file permissions for the database directory
- Try running the test script: `python test_image_reviewer.py`

### Performance Issues
- For large image directories, the tool loads images in batches
- Consider organizing images into smaller subdirectories if performance is slow
- Close other applications if memory usage is high

## Technical Details

### Database Schema
The tool uses the `image_labels` table with the following structure:
```sql
CREATE TABLE image_labels (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    image_path TEXT UNIQUE NOT NULL,
    label TEXT NOT NULL,
    confidence FLOAT,
    reviewer_notes TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_training_data BOOLEAN DEFAULT 1
);
```

### File Support
- Supported formats: `.jpg`, `.jpeg`, `.png`
- Images are automatically scaled to fit the display
- Original files are never modified

### Performance
- Loads up to 100 images per batch for memory efficiency
- Background processing of labels to maintain UI responsiveness
- Efficient database queries to avoid loading already-labeled images

## Future Enhancements

Potential improvements for future versions:
- Confidence scoring for reviewer labels
- Bulk operations (select multiple images)
- Export functionality for labeled datasets
- Integration with active learning pipelines
- Multi-reviewer support with consensus tracking 