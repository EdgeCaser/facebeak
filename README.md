# facebeak

facebeak is an AI-powered tool for identifying and tracking individual crows (and other birds) in video footage. It uses computer vision models to detect birds, assign persistent visual IDs, and optionally build a database of known individuals for long-term study.

## System Overview

The facebeak system consists of three main steps that work together to identify and track individual crows:

1. **Data Extraction** (`extract_training_data.py`)
   - Processes input videos to detect and extract individual crow images
   - Uses Faster R-CNN and YOLOv8 models to identify birds in each frame
   - Saves high-quality crow crops to the `crow_crops` directory
   - Each crow gets its own subdirectory with multiple images from different frames
   - This step is essential for building a diverse training dataset

2. **Model Training** (`train_triplet_resnet.py`)
   - Trains a ResNet-18 model using triplet loss to learn crow visual identities
   - Uses the extracted crow crops to teach the model to recognize individual crows
   - Learns to make similar crows look similar and different crows look different in the embedding space
   - Outputs a trained model file (`crow_resnet_triplet.pth`) that can identify individual crows
   - The model can be retrained with new data to improve recognition accuracy

3. **Video Processing** (`main.py` and `tracking.py`)
   - Processes new videos to detect, track, and identify individual crows
   - Uses the trained model to assign consistent IDs to crows across frames
   - Maintains a database of known crows and their sighting history
   - Outputs annotated videos with crow IDs and tracking information
   - Can be run through the GUI (`gui_launcher.py`) or command line

To improve the system's accuracy:
1. Add new videos to your collection
2. Run the extraction script to gather more crow images
3. Retrain the model with the expanded dataset
4. Process new videos with the improved model

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

## Security & Privacy

The system maintains a database (`crow_embeddings.db`) containing sensitive information about crow sightings, including:
- Visual embeddings of individual crows
- Timestamps and locations of sightings
- Video paths and frame numbers
- Confidence scores for identifications

### Database Encryption

The database is automatically encrypted for security:
- A secure random password is generated on first run
- The password is stored in a `.env` file (excluded from version control)
- The database is encrypted at rest and only decrypted during use
- Automatic backups are created during encryption
- The database is automatically re-encrypted when the program exits

### Data Protection

To protect your crow data:
- Be careful when sharing the `.env` file or database file - some data could be private
- Keep backups of both the database and `.env` file
- The database and its backups are excluded from version control 

### Publishing Data

When publishing data or results:
- Ensure you have necessary permits for wildlife observation
- Consider privacy implications
- Use aggregated data when possible
- Remove or anonymize sensitive location data
- Follow local wildlife protection guidelines

## Roadmap
- Audio analysis and UV support (planned)
- Improved crow re-identification and personality profiling
