# facebeak

facebeak is an AI-powered tool for identifying and tracking individual crows (and other birds) in video footage. It uses computer vision models to detect birds, assign persistent visual IDs, and optionally build a database of known individuals for long-term study.

## Features
- Detects crows and other birds in outdoor videos using pretrained computer vision models (Faster R-CNN, ResNet18)
- Assigns visual embeddings and matches individuals over time
- Supports persistent crow history with SQLite database
- Configurable detection and tracking thresholds
- Progress bars and debug output for transparency

## Requirements
- Python 3.8+
- See `requirements.txt` for dependencies

## Usage
1. Place your input video (e.g., `sample.mp4`) in the project directory.
2. Run the following command:

```bash
python main.py --video sample.mp4 --output output.mp4 --detection-threshold 0.3 --similarity-threshold 0.75 --skip 1
```

- `--video`: Path to input video
- `--output`: Path to save output video
- `--detection-threshold`: Detection confidence threshold (lower = more sensitive)
- `--similarity-threshold`: Visual similarity threshold for tracking (lower = more tolerant)
- `--skip`: Frame skip interval (1 = every frame)

3. The output video will be saved with bounding boxes and persistent crow IDs.
4. The system will build a database (`crow_embeddings.db`) to remember crows across sessions.

## Security & Publishing
- Before publishing, ensure all sensitive or risky files are excluded from version control (see `.gitignore`)

## Roadmap
- Audio analysis and UV support (planned)
- Improved crow re-identification and personality profiling
