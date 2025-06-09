# Facebeak Project Structure

This document describes the organized file structure of the Facebeak project.

## Directory Structure

### `/gui/` - Graphical User Interfaces
All GUI applications and interactive tools:
- `batch_image_reviewer.py` - Main image labeling interface
- `apply_model_gui.py` - GUI for applying trained models
- `train_crow_classifier_gui.py` - Training interface with real-time metrics
- `extract_training_gui.py` - GUI for extracting training data from videos
- `kivy_*.py` - Alternative Kivy-based GUI implementations
- `unsupervised_gui_tools.py` - Tools for unsupervised learning workflows

### `/utilities/` - Core Utilities and Scripts
Production-ready utilities and commonly used scripts:
- `extract_crops_cli.py` - Command-line crop extraction
- `extract_crops_cli_memory_optimized.py` - Memory-efficient version
- `train_crow_classifier.py` - Core training script
- `apply_model_to_unlabeled.py` - Batch model application
- `evaluate_model_fixed.py` - Model evaluation utilities
- `cleanup_training_output.py` - Training cleanup utilities
- `color_normalization.py` - Image preprocessing utilities
- `extract_training_data.py` - Training data preparation
- `pragmatic_detection_runner.py` - Detection pipeline runner

### `/tools/` - Development Tools and One-off Scripts
Development utilities, debugging tools, and migration scripts:
- `debug_*.py` - Various debugging utilities
- `test_*.py` - Test scripts
- `cleanup_orphaned_labels.py` - Database cleanup tool
- `consolidate_crops.py` - Data migration tool
- `crop_migration_tool.py` - Crop directory migration
- `emergency_recovery.py` - Data recovery utilities
- `browse_ec2_images.py` - Remote image browsing
- `png_to_ico.py` - Icon conversion utility

### Root Directory - Core System Files
Essential system components that other modules depend on:
- `main.py` - Main video processing pipeline
- `facebeak.py` - Core application entry point
- `detection.py` - Bird detection algorithms
- `tracking.py` - Crow tracking and ID assignment
- `db.py` - Database interface and management
- `dataset.py` - Dataset handling and loading
- `model.py` - Neural network model definitions
- `audio.py` - Audio processing components
- `crow_clustering.py` - Clustering algorithms
- `suspect_lineup.py` - Crow identification interface

## Usage Guidelines

### Running GUI Applications
```bash
# From project root
python gui/batch_image_reviewer.py
python gui/train_crow_classifier_gui.py
python gui/apply_model_gui.py
```

### Running Utilities
```bash
# From project root
python utilities/extract_crops_cli.py --help
python utilities/train_crow_classifier.py --help
python utilities/apply_model_to_unlabeled.py --help
```

### Running Tools
```bash
# From project root
python tools/debug_labels.py
python tools/cleanup_orphaned_labels.py --help
```

## Import Guidelines

When importing from these directories in your code:
```python
# For utilities
from utilities.train_crow_classifier import train_model
from utilities.extract_crops_cli import extract_crops

# For GUI components (if needed)
from gui.batch_image_reviewer import BatchImageReviewer

# For tools (rarely needed in production code)
from tools.debug_labels import debug_labels
```

## File Organization Principles

1. **GUI** - Interactive applications with user interfaces
2. **Utilities** - Reusable, production-ready scripts and modules
3. **Tools** - Development aids, debugging, and one-time migration scripts
4. **Root** - Core system components that form the foundation of the application

This structure makes it easier to:
- Find the right tool for the job
- Understand what each component does
- Maintain and update the codebase
- Onboard new developers 