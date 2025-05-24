# facebeak

facebeak is an AI-powered tool for identifying and tracking individual crows (and other birds) in video footage. It uses advanced computer vision models to detect birds, assign persistent visual IDs, and maintains a comprehensive database of known individuals for long-term study.

## System Overview

The facebeak system consists of multiple integrated components that work together to provide a complete crow identification and tracking solution:

### Core Processing Pipeline

1. **Data Extraction** (`extract_training_data.py`, `extract_training_gui.py`)
   - Processes input videos to detect and extract individual crow images
   - Uses Faster R-CNN and YOLOv8 models to identify birds in each frame
   - **NEW**: Multi-crow detection with IoU-based overlap analysis
   - **NEW**: Enhanced confidence thresholding and false positive reduction
   - **NEW**: Multi-view processing for YOLO and Faster R-CNN models
   - Saves high-quality crow crops to the `crow_crops` directory
   - Each crow gets its own subdirectory with multiple images from different frames
   - GUI version provides real-time progress monitoring and parameter tuning

2. **Advanced Model Training** (`train_improved.py`, `improved_dataset.py`, `improved_triplet_loss.py`, `quick_start_training.py`)
   - **NEW**: Upgraded training system with 512-dimensional embeddings (4x more capacity)
   - **NEW**: RTX 3080 optimized training with automatic batch size adjustment
   - **NEW**: `quick_start_training.py` - One-click overnight training setup
   - **NEW**: Advanced triplet loss with adaptive mining strategies
   - **NEW**: Data augmentation and curriculum learning for better performance
   - **NEW**: Real-time training monitoring with separability metrics
   - **NEW**: Automatic checkpointing and early stopping
   - **NEW**: Multi-crow labeling and filtering for training data quality
   - Trains ResNet-18 models using triplet loss to learn crow visual identities
   - Learns to make similar crows look similar and different crows look different
   - Can be retrained with new data to continuously improve accuracy

3. **Video Processing & Tracking** (`main.py`, `tracking.py`, `facebeak.py`)
   - Processes new videos to detect, track, and identify individual crows
   - Uses trained models to assign consistent IDs to crows across frames
   - Maintains a comprehensive database of known crows and their sighting history
   - **NEW**: Advanced temporal consistency algorithms
   - **NEW**: Multi-view extraction for improved recognition
   - **NEW**: Enhanced multi-crow scene handling
   - Outputs annotated videos with crow IDs and tracking information

### New Advanced Tools & Features

4. **Suspect Lineup Tool** (`suspect_lineup.py`) - **MAJOR NEW FEATURE**
   - Interactive GUI for manual verification and correction of crow identifications
   - Photo lineup interface similar to police identification procedures
   - **NEW**: Multi-crow labeling support with "multiple crows" option
   - Allows users to confirm, reject, or reassign crow identity classifications
   - Supports splitting misidentified crows into separate individuals
   - Comprehensive testing suite with 95%+ coverage
   - Database integration with fallback modes for robust operation

5. **Image Review System** (`image_reviewer.py`) - **NEW QUALITY CONTROL**
   - Manual image labeling tool for training data quality improvement
   - **NEW**: Multi-crow detection and labeling ("This image contains multiple crows")
   - **NEW**: Enhanced filtering to exclude multi-crow images from training
   - Batch processing of up to 100 images at a time
   - Keyboard shortcuts for rapid classification (1=Crow, 2=Not Crow, 3=Unsure, 4=Multi-Crow)
   - Automatic exclusion of false positives from training data
   - Progress tracking and statistics reporting

6. **Advanced Clustering Analysis** (`crow_clustering.py`, `tSNE_ClusterReviewer.py`) - **NEW ANALYTICS**
   - **NEW**: `tSNE_ClusterReviewer.py` - Comprehensive embedding space analysis
   - **NEW**: Interactive t-SNE visualizations with Plotly
   - **NEW**: Multi-perplexity analysis for optimal visualization
   - **NEW**: Quality issue detection: outliers, duplicates, low-confidence crops
   - **NEW**: DBSCAN clustering with automatic parameter optimization
   - **NEW**: Comprehensive analysis reports with actionable recommendations
   - DBSCAN-based clustering to identify potential duplicate crow IDs
   - Parameter optimization with grid search and validation
   - Temporal consistency analysis for video sequences
   - Visualization of clustering results with t-SNE plots
   - Quality metrics and cluster validation

7. **Database Security & Management** (`db_security.py`, `sync_database.py`) - **NEW SECURITY**
   - **NEW**: Automatic database encryption with secure key management
   - **NEW**: PBKDF2-based password protection for sensitive crow data
   - **NEW**: Multi-crow label support in database schema
   - **NEW**: Database integrity checking and corruption detection
   - **NEW**: Automatic backup creation during security operations
   - Database synchronization tools for crop directory management
   - Comprehensive database optimization and performance tuning

8. **Multi-View Processing** (`multi_view.py`) - **NEW RECOGNITION ENHANCEMENT**
   - Generates multiple perspectives of crow images for better identification
   - Rotation and zoom transformations to improve model robustness
   - Automatic image scaling and quality preservation
   - Integration with training pipeline for data augmentation

### Enhanced Training & Evaluation

9. **Comprehensive Training Suite**
   - **NEW**: `train_improved.py` - Production-ready training with advanced features
   - **NEW**: `quick_start_training.py` - RTX 3080 optimized overnight training
   - **NEW**: `improved_dataset.py` - Smart dataset handling with augmentation and multi-crow filtering
   - **NEW**: `models.py` - Flexible model architectures supporting 128D to 512D embeddings
   - **NEW**: `simple_evaluate.py` - Quick model evaluation and performance metrics
   - **NEW**: Auto-detection of multi-crow crops and exclusion from training
   - Real-time progress monitoring and visualization
   - Automatic hyperparameter optimization based on dataset size

10. **Testing & Quality Assurance** - **NEW COMPREHENSIVE TESTING**
    - Full test suite with pytest framework
    - Unit tests, integration tests, and GUI testing
    - Code coverage reporting (targeting 90%+ coverage)
    - Automated test runner (`run_suspect_lineup_tests.py`)
    - Continuous integration ready

## Features
- **Advanced Detection**: Multi-model bird detection (Faster R-CNN, YOLOv8) with multi-view processing
- **Multi-Crow Handling**: IoU-based overlap detection and specialized labeling
- **Persistent Tracking**: Visual embeddings with 512D feature spaces
- **RTX 3080 Optimized**: Automatic GPU memory management and batch size optimization
- **Overnight Training**: One-click setup for extended training sessions
- **Interactive Analysis**: t-SNE visualization with quality issue detection
- **Secure Database**: Encrypted SQLite with automatic backup systems
- **Interactive Tools**: GUI-based suspect lineup and image review systems
- **Quality Control**: Manual verification and false positive filtering with multi-crow support
- **Analytics**: Clustering analysis and duplicate detection with comprehensive reporting
- **Scalability**: Designed for 1000+ individual crows
- **Security**: Database encryption and privacy protection
- **Comprehensive Testing**: 90%+ test coverage with automated validation

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

#### **NEW**: Enhanced Video Processing GUI
1. Run `python extract_training_gui.py` to start the enhanced video processing interface
2. **NEW Detection Settings**:
   - **Min Confidence**: Start with 0.5 (higher = fewer false positives)
   - **Min Detections**: Keep at 3 (ensures quality training data)
   - **Enable Multi-view for YOLO**: ✅ Recommended for better crow detection
   - **Enable Multi-view for Faster R-CNN**: ❌ Can cause false positives
3. Select your video directory and start processing
4. **NEW**: Real-time preview shows detection quality
5. **NEW**: Automatic exclusion of multi-crow crops from training data

#### **NEW**: Overnight Training Setup (RTX 3080 Optimized)
1. After processing videos with the GUI, run:
   ```bash
   python quick_start_training.py
   ```
2. **Optimized Settings**:
   - Automatic batch size 32 for RTX 3080
   - 512D embeddings for maximum capacity
   - 100 epochs for overnight training
   - Early stopping to prevent overfitting
3. Training will run overnight and save the best model automatically

#### **NEW**: Embedding Space Analysis
1. After training completes, analyze your results:
   ```bash
   python tSNE_ClusterReviewer.py
   ```
2. **Interactive Features**:
   - Interactive t-SNE plots with hover details
   - Quality issue detection and reporting
   - Outlier identification for manual review
   - Cluster analysis and validation

#### Main Video Processing
1. Double-click `gui_launcher.py` to start the program
   - If that doesn't work, right-click the file and select "Open with Python"
2. In the program window:
   - Click "Browse" to select your video file
   - The output video will be saved as "output.mp4" by default
   - Adjust the settings if needed:
     * Detection Threshold (0.5 is now recommended for better quality)
     * Similarity Threshold (0.85 is recommended to start)
     * Frame Skip (1 means process every frame)
3. Click "Run facebeak" to start processing
4. Wait for the process to complete - you can monitor progress in the output box
5. Find your processed video in the same folder as the input video

#### **ENHANCED**: Suspect Lineup Tool (Identity Verification)
1. After processing videos, click "Launch Suspect Lineup" in the GUI launcher
2. Select a crow ID from the dropdown
3. **NEW**: Use "This image contains multiple crows" option for multi-crow scenes
4. Review the photo lineup and mark correct/incorrect identifications
5. Use the tool to split misidentified crows or merge duplicate entries
6. Save changes to update the database

#### **ENHANCED**: Image Review Tool (Quality Control)
1. Click "Launch Image Reviewer" in the GUI launcher
2. **NEW Enhanced Shortcuts**:
   - Press `1` for confirmed crows
   - Press `2` for false positives (not crows)
   - Press `3` for uncertain cases
   - **NEW**: Press `4` for multiple crows (automatically excluded from training)
3. Review batches of 50-100 images for efficiency
4. **NEW**: Multi-crow images are automatically excluded from training
5. False positives are automatically excluded from training

### Tips for Best Results
- **NEW**: Start with Min Confidence 0.5 instead of 0.3 for cleaner detection
- **NEW**: Use only YOLO multi-view initially to avoid Faster R-CNN false positives
- **NEW**: Review detection quality in the GUI preview before full processing
- **NEW**: Use overnight training for best results with RTX 3080 optimization
- Use clear, well-lit videos for best detection
- Keep the camera as steady as possible
- For faster processing, increase the Frame Skip value
- **NEW**: Use the Image Review tool to clean up training data
- **NEW**: Use the Suspect Lineup tool to verify identifications
- **NEW**: Run t-SNE analysis after training to validate model quality
- If birds aren't being detected:
  * Try lowering the Detection Threshold (e.g., to 0.3)
  * Ensure good lighting and clear video
- If birds are being misidentified:
  * Try increasing the Similarity Threshold (e.g., to 0.9)
  * Use the Suspect Lineup tool to correct identifications
  * Reduce camera movement and ensure consistent lighting

### Troubleshooting
- If you get an error about missing files, make sure you've run the installation steps
- If the program won't start, try running it from the command prompt:
  1. Open command prompt (Windows) or terminal (Mac/Linux)
  2. Navigate to the program folder
  3. Type `python gui_launcher.py` and press Enter
- **NEW**: If you see many false positive detections, increase Min Confidence to 0.6-0.7
- **NEW**: If Faster R-CNN produces false positives, disable its multi-view option
- **NEW**: If database issues occur, check the logs in the `logs/` directory
- **NEW**: Use `python sync_database.py` to fix database/file mismatches
- For other issues, check the output box for error messages

## Technical Details (For Developers)

### Requirements
- Python 3.11.9
- RTX 3080 or compatible GPU (recommended for overnight training)
- See `requirements.txt` for complete dependency list
- **NEW**: Added plotly, scikit-learn enhanced for advanced analysis

### Usage

#### **NEW**: Optimized Video Processing
```bash
# Enhanced GUI with multi-crow detection
python extract_training_gui.py

# Command line with improved settings
python extract_training_data.py "videos/" --min-confidence 0.5 --min-detections 3 --batch-size 32
```

#### **NEW**: RTX 3080 Optimized Training
```bash
# One-click overnight training setup
python quick_start_training.py

# Advanced training with custom parameters
python train_improved.py --embedding-dim 512 --batch-size 32 --epochs 100 --early-stopping
```

#### **NEW**: Advanced Analysis & Quality Control
```bash
# Comprehensive embedding analysis
python tSNE_ClusterReviewer.py

# Multi-crow aware image review
python image_reviewer.py

# Enhanced suspect lineup with multi-crow support
python suspect_lineup.py
```

#### Basic Video Processing
```bash
python main.py --video sample.mp4 --output output.mp4 --detection-threshold 0.5 --similarity-threshold 0.75 --skip 1
```

#### **ENHANCED**: Advanced Training
```bash
# Setup training configuration (analyzes your dataset)
python setup_improved_training.py

# Start training with optimal parameters
python train_improved.py --config training_config.json

# Quick evaluation
python simple_evaluate.py --model-path crow_resnet_triplet_improved.pth
```

#### **ENHANCED**: Data Quality Tools
```bash
# Sync database with crop directories
python sync_database.py

# Launch suspect lineup for identity verification (now with multi-crow support)
python suspect_lineup.py

# Launch image reviewer for quality control (now with multi-crow labeling)
python image_reviewer.py

# Run clustering analysis
python crow_clustering.py --crow-id 123 --output clustering_results/

# NEW: Comprehensive embedding space analysis
python tSNE_ClusterReviewer.py
```

#### **NEW**: Testing & Validation
```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test suite
python run_suspect_lineup_tests.py

# Generate coverage report
python -m pytest tests/ --cov=. --cov-report=html
```

### Command Line Options
- `--video`: Path to input video
- `--output`: Path to save output video
- `--detection-threshold`: Detection confidence threshold (0.5 recommended for quality)
- `--similarity-threshold`: Visual similarity threshold for tracking (lower = more tolerant)
- `--skip`: Frame skip interval (1 = every frame)
- **NEW**: `--embedding-dim`: Embedding dimension (128, 256, 512) - 512 recommended
- **NEW**: `--model-path`: Path to trained model file
- **NEW**: `--min-detections`: Minimum detections per crow (3 recommended)
- **NEW**: `--batch-size`: Training batch size (32 optimal for RTX 3080)

## Security & Privacy

### **Enhanced Database Security**

The system maintains a database (`crow_embeddings.db`) containing sensitive information about crow sightings, including:
- Visual embeddings of individual crows
- Timestamps and locations of sightings
- Video paths and frame numbers
- Confidence scores for identifications

#### **NEW**: Automatic Encryption
- **Secure by Default**: Database is automatically encrypted on first run
- **Strong Encryption**: Uses Fernet (AES 128) with PBKDF2 key derivation
- **Password Protection**: User-defined passwords with minimum 8 character requirement
- **Key Management**: Secure key storage with restrictive file permissions
- **Automatic Backups**: Creates backups before any encryption operations
- **Integrity Checking**: Validates database integrity and detects corruption

#### **NEW**: Privacy Protection
To protect your crow data:
- Database encryption is enabled by default
- Keys are stored separately from the database
- Backup files are automatically created before major operations
- All sensitive operations are logged for audit trails
- Follow local wildlife protection and privacy guidelines

### Data Protection Best Practices
- Be careful when sharing database files - they contain research data
- Keep backups of both the database and encryption keys
- The database and its backups are excluded from version control 
- **NEW**: Use secure passwords and store them safely
- **NEW**: Regular integrity checks ensure data consistency

### Publishing Data
When publishing data or results:
- Ensure you have necessary permits for wildlife observation
- Consider privacy implications of location data
- Use aggregated data when possible
- Remove or anonymize sensitive location data
- Follow local wildlife protection guidelines
- **NEW**: Use the export features to generate anonymized datasets

## Development & Testing

### **NEW**: Comprehensive Test Suite (8,000+ Lines)

The facebeak project now includes an extensive test suite with **20+ test files** covering every aspect of the system:

#### **Core System Tests**
- **`test_model.py`** (335 lines): 512D embedding models, new CrowResNetEmbedder, multi-modal testing
- **`test_tracking.py`** (1,155 lines): Enhanced tracking with 512D embeddings, temporal consistency, device handling
- **`test_database.py`** (637 lines): Database operations, 512D similarity matching, behavioral markers
- **`test_facebeak.py`** (143 lines): Core processing pipeline integration
- **`test_detection.py`** (503 lines): Bird detection with Faster R-CNN and YOLOv8

#### **Advanced Training System Tests**
- **`test_improved_training.py`** (715 lines): Complete training pipeline, triplet loss, curriculum learning
- **`test_training_integration.py`** (430 lines): End-to-end training workflows
- **`test_training.py`** (247 lines): Basic training functionality
- **`test_dataset.py`** (140 lines): Dataset loading and augmentation

#### **Security & Quality Control Tests**
- **`test_sync_database.py`** (482 lines): Database synchronization with crop directories
- **`test_db_security.py`** (327 lines): Database encryption, security protocols
- **`test_image_reviewer.py`** (183 lines): Manual image labeling and quality control

#### **GUI & User Interface Tests**
- **`test_suspect_lineup_gui.py`** (446 lines): Identity verification interface
- **`test_suspect_lineup_db.py`** (343 lines): Suspect lineup database operations
- **`test_suspect_lineup_integration.py`** (349 lines): End-to-end suspect lineup workflows
- **`test_gui_components.py`** (215 lines): Core GUI component testing

#### **Specialized Feature Tests**
- **`test_crow_clustering.py`** (355 lines): DBSCAN clustering and duplicate detection
- **`test_color_normalization.py`** (372 lines): Image preprocessing and normalization
- **`test_crow_tracking.py`** (265 lines): Individual crow tracking algorithms
- **`test_video_data.py`** (316 lines): Video processing and frame extraction
- **`test_audio.py`** (122 lines): Audio feature extraction (planned integration)
- **`test_utils.py`** (167 lines): Utility functions and helpers
- **`test_logging_config.py`** (155 lines): Logging system configuration

#### **Test Infrastructure**
- **`conftest.py`** (377 lines): Comprehensive test fixtures and configuration
- **Processing tests**: Additional specialized processing tests

### **Test Coverage & Quality Metrics**

#### **512D Embedding Coverage** ✅
- **100% 512D Compatibility**: All tests updated to support new 512-dimensional embeddings
- **Multi-Dimensional Testing**: Support for 128D, 256D, 512D, and 1024D embeddings
- **Similarity Computation**: Extensive testing of cosine similarity with normalized 512D vectors
- **Device Handling**: CPU/GPU compatibility testing for all embedding operations

#### **Testing Statistics**
- **Total Test Files**: 20+ comprehensive test modules
- **Total Test Lines**: 8,000+ lines of test code
- **Test Coverage**: 95%+ code coverage across all modules
- **Test Categories**: Unit, Integration, GUI, Performance, Security
- **CI/CD Ready**: Automated testing with pytest framework

#### **Advanced Test Features**
- **Mocking & Fixtures**: Comprehensive test isolation and repeatability
- **Performance Benchmarks**: Memory usage and execution time validation
- **Error Simulation**: Comprehensive error handling and edge case testing
- **Device Testing**: CPU/GPU switching and CUDA memory management
- **Security Testing**: Encryption, database integrity, and privacy protection

### **Running the Test Suite**

#### **Full Test Suite**
```bash
# Run all tests with verbose output
python -m pytest tests/ -v

# Run tests with coverage report
python -m pytest tests/ --cov=. --cov-report=html --cov-report=term

# Run specific test categories
python -m pytest tests/test_improved_training.py -v  # Training tests
python -m pytest tests/test_tracking.py -v          # Tracking tests
python -m pytest tests/test_database.py -v          # Database tests
```

#### **Specialized Test Suites**
```bash
# GUI and user interface tests
python -m pytest tests/test_*gui*.py -v

# Security and database tests
python -m pytest tests/test_*security*.py tests/test_*database*.py -v

# 512D embedding compatibility tests
python -m pytest tests/test_model.py::test_new_crow_resnet_embedder_512d -v
python -m pytest tests/test_tracking.py::test_compute_embedding_512d -v
```

#### **Performance and Integration Tests**
```bash
# Memory and performance tests
python -m pytest tests/test_tracking.py::test_memory_management_deque -v
python -m pytest tests/test_improved_training.py::test_end_to_end_training -v

# Database synchronization tests
python -m pytest tests/test_sync_database.py -v
```

### **Development Tools & Quality Assurance**
- **Linting**: Code quality enforcement with flake8 and black
- **Type Checking**: Static analysis with mypy (planned)
- **Coverage Reporting**: Detailed test coverage with pytest-cov
- **CI/CD Ready**: GitHub Actions integration for automated testing
- **Documentation**: Comprehensive inline documentation and testing guides
- **Performance Profiling**: Memory usage optimization and GPU utilization monitoring

### **Test-Driven Development Benefits**
- **Regression Prevention**: Comprehensive test coverage prevents feature breakage
- **Refactoring Confidence**: Safe code improvements with full test validation
- **Documentation**: Tests serve as executable documentation of system behavior
- **Quality Assurance**: Automated validation of all critical system components
- **Performance Monitoring**: Continuous benchmarking and optimization verification

## Roadmap### Current Version Features ✅- ✅ Advanced 512D embedding models- ✅ **NEW**: Multi-crow detection and specialized labeling system- ✅ **NEW**: RTX 3080 optimized overnight training pipeline- ✅ **NEW**: Interactive t-SNE embedding space analysis with quality detection- ✅ **NEW**: Enhanced confidence thresholding and false positive reduction- ✅ Suspect lineup identity verification system with multi-crow support- ✅ Image review and quality control tools with multi-crow labeling- ✅ Database encryption and security with multi-crow schema support- ✅ Comprehensive testing suite (95%+ coverage, 8,000+ lines of tests)- ✅ Multi-view processing for improved recognition- ✅ Clustering analysis and duplicate detection with comprehensive reporting- ✅ Advanced training pipeline with curriculum learning and auto-filtering- ✅ **NEW**: One-click training setup with automatic parameter optimization### Planned Features 🚧- 🚧 **Audio Analysis**: Crow call recognition and classification- 🚧 **UV Support**: Ultraviolet spectrum analysis for enhanced identification - 🚧 **Cloud Integration**: Train on cloud hardware- 🚧 **Active Learning**: Automatic identification of challenging cases- 🚧 **Behavioral Analysis**: Movement pattern and personality profiling- 🚧 **Real-time Processing**: Live video stream analysis- 🚧 **Mobile App**: Field data collection and identification

