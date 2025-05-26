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
```bash
# Extract crow images from your video files
python utilities/extract_training_gui.py
```

**What this does:**
- Scans your videos for crow detections
- Saves individual crow images to `crow_crops/` folder
- Extracts synchronized audio segments (if available)
- Creates organized folders for each detected crow

**Expected time:** 30-60 minutes per hour of video

#### Step 2: Review and Clean Data
```bash
# Launch image reviewer to clean up false detections
python image_reviewer.py
```

**What to do:**
- Press `1` for "This is a crow"
- Press `2` for "This is NOT a crow" 
- Press `3` for "Unsure"
- Press `4` for "Multiple crows in image"
- Press `q` to quit and save progress

**Expected time:** 15-30 minutes for 1000 images

#### Step 3: Setup Training Configuration
```bash
# Analyze your dataset and create optimal training settings
python utilities/setup_improved_training.py
```

**What this does:**
- Analyzes your crow image collection
- Determines optimal training parameters
- Creates `training_config.json` with recommended settings
- Provides training time estimates

### Phase 2: Model Training

#### Step 4: Train the Crow Identification Model
```bash
# Start the main training process
python train_improved.py --config training_config.json
```

**What this does:**
- Trains a neural network to recognize individual crows
- Uses triplet loss to learn visual similarities and differences
- Saves progress checkpoints every 10 epochs
- Creates visualizations of training progress

**Expected time:** 
- **With GPU:** 8-12 hours for 100 epochs
- **CPU only:** 48-72 hours for 100 epochs

**⚠️ Important:** Don't close the command window during training!

#### Step 5: Monitor Training Progress
Open a new command window and run:
```bash
# Check training logs
tail -f training.log

# Or view progress plots (updates every 10 epochs)
python -c "import matplotlib.pyplot as plt; import json; 
with open('training_output_improved/metrics_history.json') as f: 
    data = json.load(f); 
plt.plot(data['train_loss']); 
plt.title('Training Loss'); 
plt.show()"
```

### Phase 3: Model Validation and Testing

#### Step 6: Evaluate Model Performance
```bash
# Test the trained model
python simple_evaluate.py --model-path training_output_improved/crow_resnet_triplet_improved.pth
```

**What this shows:**
- Model accuracy metrics
- Separability scores (higher = better)
- Same-crow vs different-crow similarity scores
- Recommendations for improvement

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
1. **In the Facebeak GUI:**
   - Click **"Launch Training Data Extractor"** (if available)
   - **OR** close Facebeak and run: `python utilities/extract_training_gui.py`

2. **In the Extraction GUI:**
   - Click **"Browse"** to select your video files
   - Set **Detection Threshold** to `0.3` (lower = more sensitive)
   - Set **YOLO Threshold** to `0.2` 
   - Check **"Extract Audio"** if you want audio analysis
   - Click **"Start Extraction"**

3. **Wait for completion:**
   - Progress bar shows current status
   - Output window shows detailed progress
   - **Don't close the window during extraction**

**Expected time:** 30-60 minutes per hour of video

### Phase 3: Data Quality Control

#### Step 3: Review Extracted Images
1. **From main Facebeak GUI:**
   - Click **"Launch Image Reviewer"**

2. **In the Image Reviewer:**
   - Images appear one by one
   - Click **"Crow"** for valid crow images
   - Click **"Not Crow"** for false detections
   - Click **"Unsure"** if you can't tell
   - Click **"Multiple Crows"** if image has multiple birds
   - Progress shows how many you've reviewed

3. **Review until satisfied:**
   - Aim to review at least 500-1000 images
   - Focus on removing obvious false positives
   - Click **"Save and Exit"** when done

**Expected time:** 15-30 minutes for 1000 images

### Phase 4: Model Training

#### Step 4: Setup Training
1. **Close Image Reviewer and return to main GUI**

2. **Open a command window and run:**
   ```bash
   python utilities/setup_improved_training.py
   ```

3. **Review the output:**
   - Note the recommended training parameters
   - Check estimated training time
   - Ensure you have enough disk space

#### Step 5: Start Training
1. **In command window:**
   ```bash
   python train_improved.py --config training_config.json
   ```

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
1. **After training completes, run:**
   ```bash
   python simple_evaluate.py --model-path training_output_improved/crow_resnet_triplet_improved.pth
   ```

2. **Review the results:**
   - **Separability > 0.3:** Good model
   - **Separability > 0.5:** Excellent model
   - **Same-crow similarity > 0.7:** Good recognition
   - **Different-crow similarity < 0.4:** Good discrimination

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