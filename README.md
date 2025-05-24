# facebeak

facebeak is an AI-powered tool for identifying and tracking individual crows (and other birds) in video footage. It uses advanced computer vision models to detect birds, assign persistent visual IDs, and maintains a comprehensive database of known individuals for long-term study.

## System Overview

The facebeak system consists of multiple integrated components that work together to provide a complete crow identification and tracking solution:

### Core Processing Pipeline

1. **Data Extraction** (`extract_training_data.py`, `extract_training_gui.py`)
   - Processes input videos to detect and extract individual crow images
   - Uses Faster R-CNN and YOLOv8 models to identify birds in each frame
   - Saves high-quality crow crops to the `crow_crops` directory
   - Each crow gets its own subdirectory with multiple images from different frames
   - GUI version provides real-time progress monitoring and parameter tuning

2. **Advanced Model Training** (`train_improved.py`, `improved_dataset.py`, `improved_triplet_loss.py`)
   - **NEW**: Upgraded training system with 512-dimensional embeddings (4x more capacity)
   - **NEW**: Advanced triplet loss with adaptive mining strategies
   - **NEW**: Data augmentation and curriculum learning for better performance
   - **NEW**: Real-time training monitoring with separability metrics
   - **NEW**: Automatic checkpointing and early stopping
   - Trains ResNet-18 models using triplet loss to learn crow visual identities
   - Learns to make similar crows look similar and different crows look different
   - Can be retrained with new data to continuously improve accuracy

3. **Video Processing & Tracking** (`main.py`, `tracking.py`, `facebeak.py`)
   - Processes new videos to detect, track, and identify individual crows
   - Uses trained models to assign consistent IDs to crows across frames
   - Maintains a comprehensive database of known crows and their sighting history
   - **NEW**: Advanced temporal consistency algorithms
   - **NEW**: Multi-view extraction for improved recognition
   - Outputs annotated videos with crow IDs and tracking information

### New Advanced Tools & Features

4. **Suspect Lineup Tool** (`suspect_lineup.py`) - **MAJOR NEW FEATURE**
   - Interactive GUI for manual verification and correction of crow identifications
   - Photo lineup interface similar to police identification procedures
   - Allows users to confirm, reject, or reassign crow identity classifications
   - Supports splitting misidentified crows into separate individuals
   - Comprehensive testing suite with 95%+ coverage
   - Database integration with fallback modes for robust operation

5. **Image Review System** (`image_reviewer.py`) - **NEW QUALITY CONTROL**
   - Manual image labeling tool for training data quality improvement
   - Batch processing of up to 100 images at a time
   - Keyboard shortcuts for rapid classification (1=Crow, 2=Not Crow, 3=Unsure)
   - Automatic exclusion of false positives from training data
   - Progress tracking and statistics reporting

6. **Advanced Clustering Analysis** (`crow_clustering.py`) - **NEW ANALYTICS**
   - DBSCAN-based clustering to identify potential duplicate crow IDs
   - Parameter optimization with grid search and validation
   - Temporal consistency analysis for video sequences
   - Visualization of clustering results with t-SNE plots
   - Quality metrics and cluster validation

7. **Database Security & Management** (`db_security.py`, `sync_database.py`) - **NEW SECURITY**
   - **NEW**: Automatic database encryption with secure key management
   - **NEW**: PBKDF2-based password protection for sensitive crow data
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
   - **NEW**: `improved_dataset.py` - Smart dataset handling with augmentation
   - **NEW**: `models.py` - Flexible model architectures supporting 128D to 512D embeddings
   - **NEW**: `simple_evaluate.py` - Quick model evaluation and performance metrics
   - Real-time progress monitoring and visualization
   - Automatic hyperparameter optimization based on dataset size

10. **Testing & Quality Assurance** - **NEW COMPREHENSIVE TESTING**
    - Full test suite with pytest framework
    - Unit tests, integration tests, and GUI testing
    - Code coverage reporting (targeting 90%+ coverage)
    - Automated test runner (`run_suspect_lineup_tests.py`)
    - Continuous integration ready

## Features
- **Advanced Detection**: Multi-model bird detection (Faster R-CNN, YOLOv8)
- **Persistent Tracking**: Visual embeddings with 512D feature spaces
- **Secure Database**: Encrypted SQLite with automatic backup systems
- **Interactive Tools**: GUI-based suspect lineup and image review systems
- **Quality Control**: Manual verification and false positive filtering
- **Analytics**: Clustering analysis and duplicate detection
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

#### Main Video Processing
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

#### **NEW**: Suspect Lineup Tool (Identity Verification)
1. After processing videos, click "Launch Suspect Lineup" in the GUI launcher
2. Select a crow ID from the dropdown
3. Review the photo lineup and mark correct/incorrect identifications
4. Use the tool to split misidentified crows or merge duplicate entries
5. Save changes to update the database

#### **NEW**: Image Review Tool (Quality Control)
1. Click "Launch Image Reviewer" in the GUI launcher
2. Use keyboard shortcuts to quickly label images:
   - Press `1` for confirmed crows
   - Press `2` for false positives (not crows)
   - Press `3` for uncertain cases
3. Review batches of 50-100 images for efficiency
4. False positives are automatically excluded from training

#### **NEW**: Advanced Training System
1. Extract training data: Use "Extract Training Data" in the GUI
2. Start improved training: Run `python train_improved.py --config training_config.json`
3. Monitor progress in real-time with generated plots and logs
4. New model will be saved as `crow_resnet_triplet_improved.pth`

### Tips for Best Results
- Use clear, well-lit videos for best detection
- Keep the camera as steady as possible
- For faster processing, increase the Frame Skip value
- **NEW**: Use the Image Review tool to clean up training data
- **NEW**: Use the Suspect Lineup tool to verify identifications
- If birds aren't being detected:
  * Try lowering the Detection Threshold (e.g., to 0.2)
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
- **NEW**: If database issues occur, check the logs in the `logs/` directory
- **NEW**: Use `python sync_database.py` to fix database/file mismatches
- For other issues, check the output box for error messages

## Technical Details (For Developers)

### Requirements
- Python 3.11.9
- See `requirements.txt` for complete dependency list
- **NEW**: Added cryptography, scikit-learn, librosa, and testing frameworks

### Usage

#### Basic Video Processing
```bash
python main.py --video sample.mp4 --output output.mp4 --detection-threshold 0.3 --similarity-threshold 0.75 --skip 1
```

#### **NEW**: Advanced Training
```bash
# Setup training configuration (analyzes your dataset)
python setup_improved_training.py

# Start training with optimal parameters
python train_improved.py --config training_config.json

# Quick evaluation
python simple_evaluate.py --model-path crow_resnet_triplet_improved.pth
```

#### **NEW**: Data Quality Tools
```bash
# Sync database with crop directories
python sync_database.py

# Launch suspect lineup for identity verification
python suspect_lineup.py

# Launch image reviewer for quality control
python image_reviewer.py

# Run clustering analysis
python crow_clustering.py --crow-id 123 --output clustering_results/
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
- `--detection-threshold`: Detection confidence threshold (lower = more sensitive)
- `--similarity-threshold`: Visual similarity threshold for tracking (lower = more tolerant)
- `--skip`: Frame skip interval (1 = every frame)
- **NEW**: `--embedding-dim`: Embedding dimension (128, 256, 512)
- **NEW**: `--model-path`: Path to trained model file

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

### **NEW**: Comprehensive Test Suite
- **Unit Tests**: Core functionality testing with 95%+ coverage
- **Integration Tests**: End-to-end workflow validation
- **GUI Tests**: User interface component testing
- **Database Tests**: Data integrity and security validation
- **Performance Tests**: Benchmarking and optimization verification

### **NEW**: Development Tools
- **Linting**: Code quality enforcement with flake8
- **Coverage**: Detailed test coverage reporting
- **CI/CD Ready**: Automated testing and deployment support
- **Documentation**: Comprehensive inline documentation and guides

### **NEW**: Performance Monitoring
- Real-time training progress visualization
- Memory usage optimization
- Database performance tuning
- Clustering analysis metrics

## Roadmap

### Current Version Features ✅
- ✅ Advanced 512D embedding models
- ✅ Suspect lineup identity verification system
- ✅ Image review and quality control tools
- ✅ Database encryption and security
- ✅ Comprehensive testing suite (90%+ coverage)
- ✅ Multi-view processing for improved recognition
- ✅ Clustering analysis and duplicate detection
- ✅ Advanced training pipeline with curriculum learning

### Planned Features 🚧
- 🚧 **Audio Analysis**: Crow call recognition and classification
- 🚧 **UV Support**: Ultraviolet spectrum analysis for enhanced identification
- 🚧 **Mobile App**: Smartphone interface for field researchers
- 🚧 **Cloud Integration**: Multi-researcher collaboration platform
- 🚧 **Active Learning**: Automatic identification of challenging cases
- 🚧 **Behavioral Analysis**: Movement pattern and personality profiling
- 🚧 **API Integration**: RESTful API for third-party applications
- 🚧 **Real-time Processing**: Live video stream analysis

### Future Enhancements 🔮
- Advanced neural architectures (Vision Transformers, CLIP)
- Multi-species support expansion
- Federated learning for privacy-preserving collaboration
- Edge device deployment (mobile/embedded systems)
- Integration with citizen science platforms
