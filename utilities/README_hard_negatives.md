# Hard Negatives Extraction Tools

This directory contains tools for extracting hard negative training examples (non-bird objects) from videos to improve crow classification models.

## Overview

The hard negatives extraction tools help create a balanced training dataset by:
1. **Avoiding bird areas**: Uses detection models to identify bird locations and avoids sampling from those regions
2. **Random sampling**: Extracts random crops from video frames that don't contain birds
3. **Quality filtering**: Only keeps visually interesting crops (avoiding blank sky, water, etc.)
4. **Database integration**: Automatically labels extracted crops as "not_a_crow" for training

## Tools

### 1. Command Line Tool: `extract_hard_negatives.py`

Extract hard negative crops from videos via command line.

#### Usage

```bash
# Extract from single video (extract all available hard negatives)
python utilities/extract_hard_negatives.py "path/to/video.mp4" --output-dir hard_negatives

# Extract from directory of videos with custom limit
python utilities/extract_hard_negatives.py "path/to/videos/" --output-dir hard_negatives --max-crops-per-video 300

# Extract and add to database automatically
python utilities/extract_hard_negatives.py "path/to/videos/" --add-to-database --label "not_a_crow"
```

#### Parameters

- `input_path`: Path to video file or directory containing videos
- `--output-dir, -o`: Output directory for extracted crops (default: "hard_negatives_output")
- `--max-crops-per-video, -c`: Maximum number of crops to extract per video (default: 200)
- `--frame-skip, -s`: Number of frames to skip between samples (default: 30)

- `--add-to-database, -d`: Add extracted crops to database with specified label
- `--label`: Label to use when adding to database (default: "not_a_crow")
- `--video-extensions`: Video file extensions to process (default: .mp4, .avi, .mov, .mkv, .MOV)

### 2. GUI Tool: `hard_negatives_extractor_gui.py`

User-friendly graphical interface for the same functionality.

#### Usage

```bash
python gui/hard_negatives_extractor_gui.py
```

#### Features

- **File/folder selection**: Browse for single videos or entire directories
- **Parameter adjustment**: Easy sliders and inputs for all extraction parameters
- **Real-time logging**: See progress and results in the GUI
- **Database integration**: Checkbox to automatically add results to database
- **Progress tracking**: Visual progress bar and status updates

## How It Works

### 1. Detection Phase
- Runs YOLO + Faster R-CNN detection on sampled video frames
- Identifies bird locations with bounding boxes
- Creates "exclusion zones" around detected birds (with padding)

### 2. Sampling Phase
- Generates random crop locations across the frame
- Filters out crops that overlap significantly with bird areas (>10% overlap)
- Ensures crops have sufficient visual variation (not blank/uniform areas)

### 3. Quality Control
- Checks standard deviation of pixel values to avoid boring crops
- Filters out very dark or very bright uniform areas
- Uses consistent 512x512 size (hardcoded to match existing pipeline)

### 4. Database Integration
- Automatically labels crops as "not_a_crow" (or custom label)
- Checks for duplicates before adding to database
- Integrates with existing labeling and training pipeline

## Training Strategy

For optimal results, use a hybrid approach:
- **~70% crows**: Positive examples from your crow dataset
- **~20% bird false positives**: Use `extract_false_positive_crops.py` for bird-like objects misclassified as crows
- **~10% hard negatives**: Use this tool for completely non-bird objects

## Example Workflow

1. **Extract false positives** (bird-like objects):
   ```bash
   python utilities/extract_false_positive_crops.py "videos/" --output false_positives --max-crops 30
   ```

2. **Extract hard negatives** (non-bird objects):
   ```bash
   python utilities/extract_hard_negatives.py "videos/" --output hard_negatives --add-to-database
   ```

3. **Review and label** using the batch image reviewer:
   ```bash
   python gui/batch_image_reviewer.py
   ```

4. **Train improved model** with balanced dataset:
   ```bash
   python utilities/train_improved.py
   ```

## Technical Details

- **Detection models**: Uses existing YOLO + Faster R-CNN pipeline
- **Crop size**: 512x512 pixels (hardcoded to match existing pipeline)
- **Overlap threshold**: 10% maximum overlap with bird areas
- **Quality threshold**: Minimum standard deviation of 10.0 for visual interest
- **Frame sampling**: Configurable skip rate (default: every 30 frames)
- **Memory efficient**: Processes videos frame-by-frame without loading entire video

## Troubleshooting

### Common Issues

1. **"No interesting crops found"**: Try reducing quality thresholds or increasing frame sampling
2. **"Too many bird detections"**: Videos with many birds will yield fewer hard negatives (expected)
3. **Database errors**: Ensure database is accessible and not corrupted
4. **Memory issues**: Reduce batch size or crop count for large videos

### Performance Tips

- Use `--frame-skip` to process fewer frames for faster extraction
- Adjust `--max-crops-per-video` based on video content (default 200 usually works well)
- Process videos in batches rather than all at once for large datasets
- Use SSD storage for faster I/O when processing many videos

## Integration with Existing Pipeline

These tools integrate seamlessly with:
- **Detection pipeline**: Uses same models and detection logic
- **Database system**: Compatible with existing labeling schema
- **Training pipeline**: Outputs standard format for training scripts
- **Review tools**: Works with batch image reviewer for manual verification

The extracted hard negatives help create more robust crow classifiers by providing diverse negative examples that challenge the model to learn better decision boundaries. 