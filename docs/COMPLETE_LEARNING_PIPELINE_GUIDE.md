# Complete Learning Model Pipeline Guide

**A Step-by-Step Guide for Running Crow Identification Model Training and Validation**

*This guide is designed for users who are new to Python and GitHub. Follow these instructions carefully to run the complete machine learning pipeline for crow identification.*

---

## 📋 Table of Contents

1. [Initial Setup (One-Time Only)](#initial-setup)
2. [Command-Line Pipeline](#command-line-pipeline)
3. [GUI Pipeline (Recommended for Beginners)](#gui-pipeline)
4. [Understanding the Results](#understanding-results)
5. [Troubleshooting](#troubleshooting)
6. [Advanced Options](#advanced-options)

---

## 🚀 Initial Setup (One-Time Only)

### Step 1: Install Python
1. **Download Python 3.11.9** from [python.org](https://www.python.org/downloads/)
2. **During installation:**
   - ✅ Check "Add Python to PATH" 
   - ✅ Check "Install for all users"
   - ✅ Choose "Customize installation"
   - ✅ Check all optional features
   - ✅ Check "Add Python to environment variables"

### Step 2: Download the Project
1. **Download the project:**
   - Click the green "Code" button on GitHub
   - Select "Download ZIP"
   - Extract to a folder like `C:\facebeak` (Windows) or `~/facebeak` (Mac/Linux)

### Step 3: Install Required Software
1. **Open Command Prompt/Terminal:**
   - **Windows:** Press `Win + R`, type `cmd`, press Enter
   - **Mac:** Press `Cmd + Space`, type `terminal`, press Enter
   - **Linux:** Press `Ctrl + Alt + T`

2. **Navigate to the project folder:**
   ```bash
   cd C:\facebeak
   # Or on Mac/Linux:
   cd ~/facebeak
   ```

3. **Install dependencies:**
   ```bash
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```
   
   **⏱️ This will take 5-15 minutes depending on your internet speed.**

### Step 4: Verify Installation
```bash
python -c "import torch; print('✅ PyTorch installed successfully')"
python -c "import cv2; print('✅ OpenCV installed successfully')"
python -c "import sklearn; print('✅ Scikit-learn installed successfully')"
```

**If you see all three ✅ messages, you're ready to proceed!**

---

## 💻 Command-Line Pipeline

*For users comfortable with typing commands*

### Phase 1: Data Extraction and Preparation

#### Step 1: Extract Training Data from Videos
The primary script for this is `utilities/extract_training_data.py`. While a GUI like `extract_training_gui.py` might exist as a wrapper, the core logic relies on the command-line tool.
Example command:
```bash
# Extract crow images from your video files
python utilities/extract_training_data.py "path/to/your/video_directory" --output-dir "crow_crops" --min-confidence 0.3 --frame-batch-size 16 --target-fps 10 --enable-audio --correct-orientation
```

**What this does:**
- Scans videos in your `video_directory` for crow detections.
- **Crop Storage**: Saves one crop image per *detected bounding box*. This means if multiple crows are detected in a single frame, multiple crop images will be saved.
- **Directory Structure**: Organizes extracted crops primarily by video source:
    - `crow_crops/videos/VIDEO_NAME/frame_XXXXXX_crop_XXX.jpg`
- **Metadata File**: Creates `crow_crops/metadata/crop_metadata.json`. This crucial file maps each crop image (e.g., `videos/VIDEO_NAME/frame_XXXXXX_crop_XXX.jpg`) to its assigned `crow_id`, the original video name, frame number, and other detection details. This file is essential for linking crops back to individual identities and their context.
- Extracts synchronized audio segments (if `--enable-audio` is used and audio is available).
- Applies automatic orientation correction if `--correct-orientation` is used.
- The `CrowTracker` module, used internally, manages crow identities and decides when to assign new IDs or match existing ones.

**Command-Line Arguments for `extract_training_data.py`:**
- `video_dir`: Path to the directory containing your video files.
- `--output-dir`: The base directory where the `crow_crops` structure (including `videos/` and `metadata/`) will be created. This directory becomes the `base_dir` for subsequent training and evaluation steps. (Default: `crow_crops`)
- `--min-confidence`: Minimum detection score to consider a detection valid. (Default: 0.2)
- `--frame-batch-size`: Number of frames to process in a batch for detection. (Default: 16, adjust based on GPU memory)
- `--target-fps`: Target frames per second to process. The script will skip frames to approximate this rate. (Default: None, process all frames subject to `--frame-skip`)
- `--frame-skip`: Number of frames to skip between processed frames if `--target-fps` is not set. (Default: 0)
- `--enable-audio`: If specified, attempts to extract audio segments corresponding to detections.
- `--correct-orientation`: If specified, attempts to auto-correct the orientation of crow crops.

**Expected time:** 30-60 minutes per hour of video, depending on settings and hardware.

#### Step 2: Review and Clean Data
```bash
# Launch image reviewer to clean up false detections and label data
python image_reviewer.py --base-dir crow_crops
# (Assuming image_reviewer.py is updated or adapted to use the new structure and DB labels)
```

**What to do:**
- **Multi-Crop Awareness**: You might encounter multiple distinct crop images extracted from the same original video frame if multiple crows were detected. Review each crop independently.
- Press `1` for "This is a crow" (good quality, single crow)
- Press `2` for "This is NOT a crow" 
- Press `3` for "Unsure"
- Press `4` for "Multiple crows in image" (label as 'multi_crow')
- Press `5` for "Bad crow image" (e.g., blurry, partial, label as 'bad_crow')
- Press `q` to quit and save progress.
- **Labeling Importance**: Accurately labeling images is crucial. Images labeled as 'multi_crow', 'not_a_crow', or 'bad_crow' in the database will be automatically excluded from the training dataset by the `ImprovedCrowTripletDataset` loader.

**Expected time:** 15-30 minutes for 1000 images.

#### Step 3: Setup Training Configuration
```bash
# Analyze your dataset and create optimal training settings
python utilities/setup_improved_training.py --base-dir crow_crops
```

**What this does:**
- Analyzes your crow image collection using `crop_metadata.json` and database labels to consider only valid training images.
- Determines optimal training parameters based on the filtered dataset.
- Creates `training_config.json` with recommended settings. This configuration file will use `base_dir` (e.g., "crow_crops") to specify the root data directory.
- Provides training time estimates.

### Phase 2: Model Training

#### Step 4: Train the Crow Identification Model
```bash
# Start the main training process
python train_improved.py --config training_config.json
```

**What this does:**
- Trains a neural network to recognize individual crows using `ImprovedCrowTripletDataset`.
- **Dataset Source**: The `ImprovedCrowTripletDataset` loads images based on the `crop_metadata.json` file (found within the `base_dir` specified in the config) and reads the actual image files from the `crow_crops/videos/` subdirectories.
- **Filtering**: The dataset loader automatically filters out images that are labeled in the database as 'multi_crow', 'not_a_crow', or 'bad_crow', ensuring only clean data is used for training.
- Uses triplet loss to learn visual similarities and differences.
- Saves progress checkpoints (e.g., every 10 epochs, configurable).
- Creates visualizations of training progress.

**Expected time:** 
- **With GPU:** 8-12 hours for 100 epochs (varies greatly with dataset size and GPU model).
- **CPU only:** 48-72+ hours for 100 epochs.

**⚠️ Important:** Don't close the command window during training!

#### Step 5: Monitor Training Progress
Open a new command window and run:
```bash
# Check training logs (path might be inside a timestamped run folder within your base_output_dir from config)
tail -f training_output/your_model_timestamp/training.log

# Or view progress plots (updates as configured)
# Example: python -c "import matplotlib.pyplot as plt; import json;
# with open('training_output/your_model_timestamp/metrics_history.json') as f:
#    data = json.load(f);
# plt.plot(data['train_loss']);
# plt.title('Training Loss');
# plt.show()"
```
*(Adjust paths to `metrics_history.json` based on your actual output directory structure defined in `training_config.json`)*

### Phase 3: Model Validation and Testing

#### Step 6: Evaluate Model Performance
Update the command to use `--base-dir` and new threshold arguments:
```bash
# Test the trained model
python utilities/simple_evaluate.py --model-path training_output/your_model_timestamp/best_model.pth --base-dir crow_crops --id-similarity-threshold 0.5 --non-crow-similarity-threshold 0.4
```

**What this shows:**
The evaluation script has been enhanced:
- **Data Loading**: It now also uses `crop_metadata.json` and database labels (from the specified `base_dir`) to load appropriate samples, including images labeled as 'not_a_crow' for comprehensive testing.
- **Crow Identification Metrics**:
    - Precision, Recall, F1-Score, Accuracy for distinguishing between *known crow IDs*.
    - Average similarity for same-crow pairs and different-crow pairs.
    - Separability score for known crow IDs.
- **Crow vs. Non-Crow Distinction Metrics (New)**:
    - `non_crow_true_rejection_rate`: How often 'not_a_crow' images are correctly identified as not matching any known crow below the `--non-crow-similarity-threshold`.
    - `non_crow_false_alarm_rate`: How often 'not_a_crow' images are mistakenly matched to a known crow above the `--non-crow-similarity-threshold`.
- The thresholds used for both types of evaluations are reported.

**Command-Line Arguments for `simple_evaluate.py`:**
- `--model-path`: Path to your trained model file (e.g., `best_model.pth` or a specific checkpoint).
- `--base-dir`: The root directory containing the `crow_crops` structure (especially `metadata/crop_metadata.json` and the `videos/` image folders).
- `--id-similarity-threshold`: Cosine similarity threshold for considering two known crow images as a match. (Default: 0.5)
- `--non-crow-similarity-threshold`: Cosine similarity threshold for deciding if a 'not_a_crow' image is distinct enough from all known crows. (Default: 0.4)
- `--max-crows`: Maximum number of unique crow individuals to sample for evaluation.
- `--max-samples-per-crow`: Maximum image samples per crow individual.

#### Step 7: Advanced Analysis (Optional)
```bash
# Run clustering analysis to find potential issues
python crow_clustering.py

# Launch suspect lineup for manual verification
python suspect_lineup.py

# Run unsupervised learning improvements
python unsupervised_workflow.py
```

### Phase 4: Deploy and Test

#### Step 8: Test on New Videos
```bash
# Process a new video with your trained model
python main.py --video "path/to/your/video.mp4" \
               --skip-output "output_skip.mp4" \
               --full-output "output_full.mp4" \
               --detection-threshold 0.3 \
               --preserve-audio
```

**What this creates:**
- `output_skip.mp4`: Video with detections on processed frames
- `output_full.mp4`: Full video with interpolated tracking
- Database entries for all detected crows

---

## 🖱️ GUI Pipeline (Recommended for Beginners)

*Point-and-click interface for all operations*

### Phase 1: Launch the Main Interface

#### Step 1: Start the GUI
```bash
python facebeak.py
```

**This opens the main Facebeak window with all tools.**

### Phase 2: Data Extraction

#### Step 2: Extract Training Data
1.  **In the Facebeak GUI (if it wraps `extract_training_data.py`):**
    *   Look for options to set the source video directory and the main output directory (this will be your `base_dir`, e.g., `crow_crops`).
    *   Configure parameters like detection confidence, target FPS, audio extraction, and orientation correction as available.
    *   Start the extraction.
    *   **Output**: The process will create the `crow_crops/videos/VIDEO_NAME/...` structure and `crow_crops/metadata/crop_metadata.json` as described in the command-line section.

2.  **If using `extract_training_gui.py` directly (assuming it's updated or a wrapper):**
    *   The GUI should allow specifying the video source and the main output directory (which becomes `base_dir`).
    *   It should reflect new parameters like target FPS, frame batch size, etc.
    *   **Crop Storage & Metadata**: Understand that it saves one crop per detection and creates the `crop_metadata.json` file.

3. **Wait for completion:**
   - Progress bar shows current status
   - Output window shows detailed progress
   - **Don't close the window during extraction**

**Expected time:** 30-60 minutes per hour of video

### Phase 3: Data Quality Control

#### Step 3: Review Extracted Images
1.  **From main Facebeak GUI (or by running `python image_reviewer.py --base-dir crow_crops`):**
    *   Launch the Image Reviewer. Ensure it's configured to use the database for storing labels.
2.  **In the Image Reviewer:**
    *   **Multi-Crop Awareness**: Remember that you might be reviewing multiple crops from the same original video frame. Each is a distinct detection.
    *   Label images using categories like:
        *   "Crow" (good, single, clear crow)
        *   "Not Crow"
        *   "Unsure"
        *   "Multiple Crows" (important for exclusion from triplet training)
        *   "Bad Crow Image" (blurry, partial, etc.)
    *   These labels are saved to the database and are critical for later dataset preparation.
3.  **Review diligently.** The quality of your dataset heavily depends on this step.
    *   Images labeled as 'multi_crow', 'not_a_crow', or 'bad_crow' (and potentially others based on dataset loader configuration) will be automatically excluded from certain training datasets (like `ImprovedCrowTripletDataset`).

**Expected time:** 15-30 minutes for 1000 images.

### Phase 4: Model Training

#### Step 4: Setup Training
1.  **Close Image Reviewer and return to main GUI (if applicable).**
2.  **Open a command window and run:**
    ```bash
    python utilities/setup_improved_training.py --base-dir crow_crops
    # This script now uses --base-dir
    ```
3.  **Review the output:**
    *   The script analyzes the data (from `crop_metadata.json` and filtered by DB labels) in your `base_dir` (e.g., `crow_crops`).
    *   It recommends training parameters and creates `training_config.json`.
    *   The `training_config.json` will now use a `base_dir` key (e.g., value: "crow_crops") for the dataset loader.

#### Step 5: Start Training
1.  **In command window:**
    ```bash
    python train_improved.py --config training_config.json
    ```
    *   The `ImprovedCrowTripletDataset` (or similar used by `train_improved.py`) will load data using `crop_metadata.json` from the `base_dir` specified in the config. It will read images from the `videos/` subdirectory and automatically filter out images based on their database labels ('multi_crow', 'not_a_crow', 'bad_crow').

2. **Monitor progress:**
   - Training logs appear in the command window
   - Progress plots are saved to `training_output_improved/`
   - Checkpoints are saved every 10 epochs

**⚠️ Important:** 
- Keep the command window open during training
- Training can take 8-72 hours depending on your hardware
- You can safely use your computer for other tasks

### Phase 5: Validation and Analysis

#### Step 6: Evaluate Your Model
1.  **After training completes, run (note the updated arguments):**
    ```bash
    python utilities/simple_evaluate.py --model-path training_output/your_model_timestamp/best_model.pth --base-dir crow_crops --id-similarity-threshold 0.5 --non-crow-similarity-threshold 0.4
    ```
    *(Adjust model path and thresholds as needed)*

2.  **Review the results:**
    *   The script now uses `crop_metadata.json` and DB labels from the `--base-dir` to load evaluation samples, including 'crow' and 'not_a_crow' images.
    *   **Crow Identification Metrics**:
        *   `precision_id`, `recall_id`, `f1_id`, `accuracy_id`: Performance on distinguishing between known crow IDs.
        *   `avg_similarity_pos_id`, `avg_similarity_neg_id`, `separability_id`: Analyze similarity scores for same vs. different known crows.
    *   **Crow vs. Non-Crow Distinction Metrics**:
        *   `non_crow_true_rejection_rate`: Percentage of 'not_a_crow' images correctly identified as not matching any known crow (below `--non-crow-similarity-threshold`).
        *   `non_crow_false_alarm_rate`: Percentage of 'not_a_crow' images incorrectly flagged as similar to a known crow.
    *   A good model will have high values for ID precision/recall/F1, good separability for ID, a high non-crow true rejection rate, and a low non-crow false alarm rate.

#### Step 7: Advanced Quality Control
1. **From main Facebeak GUI:**
   - Click **"Launch Suspect Lineup"** to manually verify identifications
   - Click **"Run Clustering"** to find potential duplicate IDs

2. **In Suspect Lineup:**
   - Review suggested crow identifications
   - Confirm or reject each suggestion
   - Split crows that were incorrectly merged

### Phase 6: Deploy and Test

#### Step 8: Process New Videos
1. **In main Facebeak GUI:**
   - Click **"Add Videos"** to select new video files
   - Adjust detection thresholds if needed:
     - **Detection Threshold:** 0.3 (lower = more sensitive)
     - **YOLO Threshold:** 0.2
     - **IOU Threshold:** 0.2 (lower = more lenient tracking)
   - Set **Output Directory** to where you want results
   - Click **"Process Videos"**

2. **Monitor processing:**
   - Progress appears in the output window
   - Processing time depends on video length and settings
   - Results are saved to your specified output directory

**Expected time:** 10-30 minutes per hour of video

---

## 📊 Understanding the Results

### Training Metrics

**Separability Score:**
- **> 0.5:** Excellent - Model can distinguish crows very well
- **0.3-0.5:** Good - Model works well for most crows
- **< 0.3:** Poor - Model needs more training or data

**Loss Values:**
- **Training Loss:** Should decrease over time
- **< 0.5:** Good convergence
- **Still decreasing:** Model is still learning

**Similarity Scores:**
- **Same-crow similarity > 0.7:** Good recognition
- **Different-crow similarity < 0.4:** Good discrimination
- **Gap between them > 0.3:** Excellent separation

### Output Files

**After Training:**
- `crow_resnet_triplet_improved.pth` - Your trained model
- `training_output_improved/` - All training results
- `training.log` - Detailed training log
- `metrics_history.json` - Performance over time

**After Video Processing:**
- `output_skip.mp4` - Video with detections on key frames
- `output_full.mp4` - Full video with tracking
- Updated database with new crow sightings

### Database Information

**Crow Database (`facebeak.db`):**
- Stores all known crows and their sightings
- Each crow gets a unique ID number
- Tracks when and where each crow was seen
- Stores visual embeddings for identification

---

## 🔧 Troubleshooting

### Common Issues and Solutions

#### "Python not found" or "pip not found"
**Solution:**
1. Reinstall Python with "Add to PATH" checked
2. Restart your command prompt
3. Try `python3` instead of `python`

#### "CUDA out of memory" during training
**Solution:**
```bash
# Reduce batch size
python train_improved.py --batch-size 16 --config training_config.json
```

#### Training is very slow
**Solutions:**
1. **Reduce dataset size temporarily:**
   ```bash
   python train_improved.py --epochs 20 --config training_config.json
   ```

2. **Use smaller embedding dimension:**
   ```bash
   python train_improved.py --embedding-dim 256 --config training_config.json
   ```

#### "No module named 'torch'" or similar
**Solution:**
```bash
pip install --upgrade torch torchvision torchaudio
```

#### GUI windows don't open
**Solution:**
1. **On Linux, install tkinter:**
   ```bash
   sudo apt-get install python3-tk
   ```

2. **Try running with Python directly:**
   ```bash
   python -m tkinter
   ```

#### Training stops unexpectedly
**Solution:**
```bash
# Resume from last checkpoint
python train_improved.py --resume-from training_output_improved/checkpoints/latest_checkpoint.pth
```

#### Poor model performance
**Solutions:**
1. **Review more images in Image Reviewer**
2. **Increase training epochs:**
   ```bash
   python train_improved.py --epochs 150 --config training_config.json
   ```
3. **Check for data quality issues:**
   ```bash
   python crow_clustering.py
   ```

### Getting Help

**Check logs for errors:**
- `training.log` - Training issues
- `facebeak_session.log` - Video processing issues
- Command window output - Real-time errors

**Common log messages:**
- `"CUDA available: True"` - GPU acceleration working
- `"Epoch X/Y completed"` - Training progressing normally
- `"Early stopping triggered"` - Training finished automatically
- `"Checkpoint saved"` - Progress saved successfully

---

## ⚙️ Advanced Options

### Custom Training Parameters

**For larger datasets (1000+ crows):**
```bash
python train_improved.py --embedding-dim 512 --epochs 200 --batch-size 32
```

**For faster training (less accuracy):**
```bash
python train_improved.py --embedding-dim 128 --epochs 50 --batch-size 64
```

**For maximum accuracy (slower):**
```bash
python train_improved.py --embedding-dim 512 --epochs 300 --learning-rate 0.0001
```

### Unsupervised Learning Enhancement

**Run advanced unsupervised techniques:**
```bash
# Complete unsupervised workflow
python unsupervised_workflow.py

# Or individual components:
python -c "
from unsupervised_learning import UnsupervisedTrainingPipeline
from models import CrowResNetEmbedder
import torch

model = CrowResNetEmbedder()
pipeline = UnsupervisedTrainingPipeline(model, {})
results = pipeline.apply_unsupervised_techniques('your_video.mp4')
print('Unsupervised analysis complete!')
"
```

### Batch Processing Multiple Videos

**Process many videos at once:**
```bash
# Create a batch script
echo "python main.py --video video1.mp4 --skip-output out1_skip.mp4 --full-output out1_full.mp4" > process_all.bat
echo "python main.py --video video2.mp4 --skip-output out2_skip.mp4 --full-output out2_full.mp4" >> process_all.bat
echo "python main.py --video video3.mp4 --skip-output out3_skip.mp4 --full-output out3_full.mp4" >> process_all.bat

# Run the batch
process_all.bat
```

### Performance Optimization

**For RTX 3080 or similar GPU:**
```bash
python train_improved.py --batch-size 64 --num-workers 8 --config training_config.json
```

**For older/slower hardware:**
```bash
python train_improved.py --batch-size 16 --num-workers 2 --config training_config.json
```

---

## 🎯 Success Checklist

### ✅ Phase 1: Setup Complete
- [ ] Python 3.11.9 installed with PATH
- [ ] All dependencies installed without errors
- [ ] Verification commands show ✅ for all packages
- [ ] Project folder accessible from command line

### ✅ Phase 2: Data Ready
- [ ] Videos processed and crow images extracted
- [ ] At least 500 images reviewed in Image Reviewer
- [ ] False positives removed from training data
- [ ] Training configuration created successfully

### ✅ Phase 3: Training Complete
- [ ] Training ran for at least 50 epochs without errors
- [ ] Final separability score > 0.3
- [ ] Model file saved successfully
- [ ] Training plots show decreasing loss

### ✅ Phase 4: Validation Passed
- [ ] Model evaluation shows good metrics
- [ ] Test video processing works correctly
- [ ] Database contains detected crows
- [ ] Output videos show proper tracking

### ✅ Phase 5: Production Ready
- [ ] Suspect lineup tool works for verification
- [ ] Clustering analysis identifies potential issues
- [ ] New videos process successfully
- [ ] Results meet your accuracy requirements

---

**🎉 Congratulations! You've successfully completed the full machine learning pipeline for crow identification!**

Your system is now ready to identify and track individual crows in new videos. The model will continue to improve as you add more data and refine the training process.

For ongoing support and advanced features, refer to the other documentation files in the `docs/` folder. 