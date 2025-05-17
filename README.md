# facebeak

facebeak is an AI-powered tool for identifying and tracking individual crows (and other birds) in video footage. It uses computer vision models to detect birds, assign persistent visual IDs, and optionally build a database of known individuals for long-term study.

## Features
- Detects crows and other birds in outdoor videos using pretrained computer vision models (Faster R-CNN, ResNet18)
- Assigns visual embeddings and matches individuals over time
- Supports persistent crow history with SQLite database
- Configurable detection and tracking thresholds
- Progress bars and debug output for transparency

## Quick Start Guide (For Non-Coders)

### Installation
1. Download and install Python 3.11.9 
   - During installation, make sure to check "Add Python to PATH"
2. Download this project by clicking the green "Code" button above and selecting "Download ZIP"
3. Extract the ZIP file to a folder on your computer
4. Open a command prompt (Windows) or terminal (Mac/Linux) in the extracted folder
5. Run these commands to set up the required software:
   ```
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

### Using the Program
1. Double-click `gui_launcher.py` to start the program
   - If that doesn't work, right-click the file and select "Open with Python"
2. In the program window:
   - Click "Browse" to select your video file
   - The output video will be saved as "output.mp4" by default
   - Adjust the settings if needed:
     * Detection Threshold (0.3 is recommended to start)
     * Similarity Threshold (0.85 is recommended to start)
     * Frame Skip (1 means process every frame)
3. Click "Run facebeak" to start processing
4. Wait for the process to complete - you can monitor progress in the output box
5. Find your processed video in the same folder as the input video

### Tips for Best Results
- Use clear, well-lit videos for best detection
- Keep the camera as steady as possible
- For faster processing, increase the Frame Skip value
- If birds aren't being detected:
  * Try lowering the Detection Threshold (e.g., to 0.2)
  * Ensure good lighting and clear video
- If birds are being misidentified:
  * Try increasing the Similarity Threshold (e.g., to 0.9)
  * Reduce camera movement
  * Ensure consistent lighting

### Troubleshooting
- If you get an error about missing files, make sure you've run the installation steps
- If the program won't start, try running it from the command prompt:
  1. Open command prompt (Windows) or terminal (Mac/Linux)
  2. Navigate to the program folder
  3. Type `python gui_launcher.py` and press Enter
- For other issues, check the output box for error messages

## Technical Details (For Developers)

## Requirements
- Python 3.11.9
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
