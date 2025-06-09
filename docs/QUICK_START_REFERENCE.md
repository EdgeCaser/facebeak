# Quick Start Reference Card

**Essential Commands for Crow Identification Pipeline**

---

## 🚀 One-Time Setup

```bash
# 1. Install Python 3.11.9 from python.org (check "Add to PATH")
# 2. Download project and extract to folder
cd C:\facebeak  # or your project folder
python -m pip install --upgrade pip
pip install -r requirements.txt

# 3. Verify installation
python -c "import torch; print('✅ Ready!')"
```

---

## 💻 Command-Line Quick Start

### Complete Pipeline (4 Steps)

```bash
# Step 1: Extract training data
python utilities/extract_training_gui.py

# Step 2: Clean data (press 1=crow, 2=not crow, q=quit)
python image_reviewer.py

# Step 3: Setup training
python utilities/setup_improved_training.py

# Step 4: Train model (8-72 hours)
python train_improved.py --config training_config.json
```

### Test Your Model

```bash
# Evaluate performance
python simple_evaluate.py --model-path training_output_improved/crow_resnet_triplet_improved.pth

# Process new video
python main.py --video "input.mp4" --skip-output "out_skip.mp4" --full-output "out_full.mp4"
```

---

## 🖱️ GUI Quick Start

### Main Interface

```bash
python facebeak.py
```

### Training Data Extraction

```bash
python utilities/extract_training_gui.py
```

### Essential GUI Steps

1. **Extract Data:** Run extraction GUI → Browse videos → Start extraction
2. **Clean Data:** Launch Image Reviewer → Review images → Save and exit
3. **Train Model:** Run training command in terminal (see above)
4. **Process Videos:** Main GUI → Add videos → Process videos

---

## 🔧 Common Issues & Quick Fixes

### Python/Installation Issues

```bash
# Python not found
python3 --version  # Try python3 instead

# Missing packages
pip install torch torchvision opencv-python scikit-learn

# GUI won't open (Linux)
sudo apt-get install python3-tk
```

### Training Issues

```bash
# Out of memory
python train_improved.py --batch-size 16 --config training_config.json

# Too slow
python train_improved.py --epochs 20 --embedding-dim 256

# Resume training
python train_improved.py --resume-from training_output_improved/checkpoints/latest_checkpoint.pth
```

### Performance Issues

```bash
# Check GPU
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Reduce dataset temporarily
python train_improved.py --epochs 50 --config training_config.json
```

---

## 📊 Success Metrics

### Good Training Results

- **Separability > 0.3** (Good) or **> 0.5** (Excellent)
- **Training loss < 0.5** after convergence
- **Same-crow similarity > 0.7**
- **Different-crow similarity < 0.4**

### File Locations

- **Trained model:** `training_output_improved/crow_resnet_triplet_improved.pth`
- **Training logs:** `training.log`
- **Crow images:** `crow_crops/`
- **Database:** `facebeak.db`

---

## ⚡ Advanced Quick Commands

### Clustering & Analysis

```bash
python crow_clustering.py              # Find duplicate IDs
python suspect_lineup.py               # Manual verification
python unsupervised_workflow.py        # Advanced improvements
```

### Batch Processing

```bash
# Process multiple videos
for video in *.mp4; do
    python main.py --video "$video" --skip-output "${video%.*}_skip.mp4" --full-output "${video%.*}_full.mp4"
done
```

### Custom Training

```bash
# Large dataset (1000+ crows)
python train_improved.py --embedding-dim 512 --epochs 200 --batch-size 32

# Fast training (less accuracy)
python train_improved.py --embedding-dim 128 --epochs 50 --batch-size 64

# Maximum accuracy
python train_improved.py --embedding-dim 512 --epochs 300 --learning-rate 0.0001
```

---

## 📞 Getting Help

### Check Logs

- `training.log` - Training issues
- `facebeak_session.log` - Video processing
- Command window - Real-time errors

### Key Log Messages

- `"CUDA available: True"` - GPU working ✅
- `"Epoch X/Y completed"` - Training normal ✅
- `"Early stopping triggered"` - Training done ✅
- `"Checkpoint saved"` - Progress saved ✅

---

**💡 Tip:** Keep this reference open while following the complete guide in `COMPLETE_LEARNING_PIPELINE_GUIDE.md`
