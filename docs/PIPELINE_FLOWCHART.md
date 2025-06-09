# Learning Pipeline Flowchart

**Visual Guide to the Complete Crow Identification Pipeline**

---

## 📊 Complete Pipeline Overview

```
┌─────────────────┐
│   START HERE    │
│  Install Python │
│  & Dependencies │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│  PHASE 1: DATA  │
│   EXTRACTION    │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐    ┌─────────────────┐
│  Command Line   │    │   GUI Version   │
│                 │    │                 │
│ extract_training│◄──►│extract_training │
│     _gui.py     │    │     _gui.py     │
└─────────┬───────┘    └─────────┬───────┘
          │                      │
          └──────────┬───────────┘
                     │
                     ▼
          ┌─────────────────┐
          │  PHASE 2: DATA  │
          │   CLEANING      │
          └─────────┬───────┘
                    │
                    ▼
          ┌─────────────────┐
          │ image_reviewer  │
          │      .py        │
          │                 │
          │ 1=Crow 2=Not    │
          │ 3=Unsure 4=Multi│
          └─────────┬───────┘
                    │
                    ▼
          ┌─────────────────┐
          │  PHASE 3: SETUP │
          │   TRAINING      │
          └─────────┬───────┘
                    │
                    ▼
          ┌─────────────────┐
          │setup_improved_  │
          │   training.py   │
          │                 │
          │Creates config   │
          └─────────┬───────┘
                    │
                    ▼
          ┌─────────────────┐
          │ PHASE 4: MODEL  │
          │    TRAINING     │
          └─────────┬───────┘
                    │
                    ▼
          ┌─────────────────┐
          │ train_improved  │
          │      .py        │
          │                 │
          │ 8-72 hours      │
          └─────────┬───────┘
                    │
                    ▼
          ┌─────────────────┐
          │ PHASE 5: MODEL  │
          │   VALIDATION    │
          └─────────┬───────┘
                    │
                    ▼
          ┌─────────────────┐
          │simple_evaluate  │
          │      .py        │
          │                 │
          │Check metrics    │
          └─────────┬───────┘
                    │
                    ▼
          ┌─────────────────┐
          │ PHASE 6: DEPLOY │
          │   & TEST        │
          └─────────┬───────┘
                    │
                    ▼
┌─────────────────┐    ┌─────────────────┐
│  Command Line   │    │   GUI Version   │
│                 │    │                 │
│    main.py      │◄──►│   facebeak.py   │
│                 │    │                 │
│Process videos   │    │Process videos   │
└─────────────────┘    └─────────────────┘
```

---

## 🔄 Detailed Phase Breakdown

### Phase 1: Data Extraction (30-60 min per video hour)
```
Input Videos ──► Detection Models ──► Crow Crops & Metadata ──► Audio Segments
     │                  │                    │                       │
     │            ┌─────────────┐            │                       │
     │            │ Faster R-CNN│            │                       │
     │            │   YOLOv8    │            │                       │
     │            └─────────────┘            │                       │
     │                                       │                       │
     └───────────────────────────────────────┼───────────────────────┘
                                             │
                                             ▼
                                 crow_crops/
                                 ├── videos/VIDEO_NAME/frame_XXXXXX_crop_XXX.jpg (One crop per detection)
                                 ├── metadata/crop_metadata.json (Maps crops to crow_id, frame, video)
                                 └── audio/ (If extracted)
```
*Key Change Note: System now saves one crop image per detection, not just one per frame.*

### Phase 2: Data Cleaning (15-30 min per 1000 images)
```
Raw Crow Crops ──► Manual Review (ImageReviewer) ──► Labeled Data in DB ──► Clean Dataset for Training
       │                  │                                  │                       │
   ┌───────────┐    ┌─────────────┐                  ┌───────────────┐          ┌─────────────┐
   │ All crops │    │1=Crow       │                  │ Labels stored │          │ Verified    │
   │ (from     │───►│2=Not_a_crow │─────────────────►│ in database   │─────────►│ crow images │
   │ videos/)  │    │3=Unsure     │                  │ (facebeak.db) │          │ (for model) │
   └───────────┘    │4=Multi_crow │                  └───────────────┘          └─────────────┘
                    │5=Bad_crow   │
                    └─────────────┘
```
*Note: Labels ('not_a_crow', 'multi_crow', 'bad_crow') stored in the database are crucial for filtering during dataset preparation.*

### Phase 3: Training Setup (5 min)
```
Clean Dataset (via DB labels & metadata) ──► Analysis ──► Configuration ──► Ready to Train
              │                                  │             │                  │
  ┌──────────────────────────┐             ┌─────────┐   ┌─────────────┐   ┌─────────────┐
  │ Uses crop_metadata.json  │             │Optimal  │   │training_    │   │ Estimated   │
  │ & DB labels to select    │────────────►│batch    │──►│config.json  │──►│ time &      │
  │ valid training images    │             │size etc │   │(uses base_dir)│   │ resources   │
  └──────────────────────────┘             └─────────┘   └─────────────┘   └─────────────┘
```

### Phase 4: Model Training (8-72 hours)
```
Training Data (from ImprovedCrowTripletDataset) ──► Neural Network ──► Trained Model
                  │                                       │                │
  ┌───────────────────────────────────┐      ┌─────────────┐   ┌─────────────┐
  │ - Loads images via crop_metadata  │      │ ResNet-18   │   │ Embeddings  │
  │ - Reads from videos/ dir          │─────►│ + Triplet   │──►│ model       │
  │ - Filters using DB labels         │      │ Loss        │   │ (.pth file) │
  │   (excludes 'multi_crow', etc.)   │      └─────────────┘   └─────────────┘
  └───────────────────────────────────┘            │
                                                   ▼
                                             ┌─────────────┐
                                             │ Checkpoints │
                                             │ (e.g. every │
                                             │ 10 epochs)  │
                                             └─────────────┘
```

### Phase 5: Validation (5-10 min)
```
Trained Model ──► Test Data (from metadata & DB) ──► Performance Metrics
      │                        │                            │
  ┌─────────┐   ┌───────────────────────────────┐    ┌──────────────────────────┐
  │ .pth    │   │ Loads crow & 'not_a_crow'     │    │ - Crow ID Metrics        │
  │ file    │──►│ samples via crop_metadata.json│───►│   (Precision, Recall)    │
  │         │   │ and DB labels.                │    │ - Crow vs. Non-Crow      │
  └─────────┘   └───────────────────────────────┘    │   (Rejection/Alarm Rates)│
                                                     └──────────────────────────┘
```

### Phase 6: Deployment (10-30 min per video hour)
```
New Videos ──► Detection ──► Tracking ──► Identification ──► Results
     │            │            │             │               │
 ┌─────────┐ ┌─────────┐ ┌─────────────┐ ┌─────────┐  ┌─────────────┐
 │ Raw     │ │ Bird    │ │ Consistent  │ │ Match   │  │ Annotated   │
 │ footage │►│ bboxes  │►│ IDs across  │►│ to known│─►│ videos +    │
 │         │ │         │ │ frames      │ │ crows   │  │ database    │
 └─────────┘ └─────────┘ └─────────────┘ └─────────┘  └─────────────┘
```

---

## 🎯 Decision Points & Branching

### Quality Control Checkpoints
```
After Data Extraction:
├── Good quality images (>80% crows) ──► Continue to training
├── Mixed quality (50-80% crows) ──► Manual review required
└── Poor quality (<50% crows) ──► Adjust detection thresholds

After Training:
├── Separability > 0.5 ──► Excellent, deploy immediately
├── Separability 0.3-0.5 ──► Good, consider more training
└── Separability < 0.3 ──► Poor, need more data or review

After Validation:
├── High accuracy ──► Production ready
├── Medium accuracy ──► Run clustering analysis
└── Low accuracy ──► Return to data cleaning
```

### Troubleshooting Paths
```
Training Issues:
├── Out of memory ──► Reduce batch size ──► Continue
├── Too slow ──► Reduce epochs/dimensions ──► Continue
├── Poor convergence ──► Check data quality ──► Review/retrain
└── Crashes ──► Resume from checkpoint ──► Continue

Performance Issues:
├── False positives ──► Adjust thresholds ──► Reprocess
├── Missed detections ──► Lower thresholds ──► Reprocess
├── ID confusion ──► Run suspect lineup ──► Manual correction
└── Poor tracking ──► Adjust IOU settings ──► Reprocess
```

---

## ⏱️ Time Estimates by Phase

### Typical Timeline for 1000 Crow Dataset
```
Phase 1: Data Extraction     │████████████░░░░░░░░░░│ 2-4 hours
Phase 2: Data Cleaning       │██░░░░░░░░░░░░░░░░░░░░│ 30 min
Phase 3: Training Setup      │░░░░░░░░░░░░░░░░░░░░░░│ 5 min
Phase 4: Model Training      │████████████████████░│ 8-72 hours
Phase 5: Validation          │░░░░░░░░░░░░░░░░░░░░░░│ 10 min
Phase 6: Deployment          │██░░░░░░░░░░░░░░░░░░░░│ Variable

Total: 11-80 hours (mostly automated)
```

### Resource Requirements
```
CPU Training:    │████████████████████│ 48-72 hours
GPU Training:    │████░░░░░░░░░░░░░░░░│ 8-12 hours
Memory Usage:    │██████░░░░░░░░░░░░░░│ 8-16 GB RAM
Disk Space:      │████░░░░░░░░░░░░░░░░│ 10-50 GB
```

---

## 🚀 Quick Start Paths

### For Beginners (GUI Path)
```
1. facebeak.py ──► 2. extract_training_gui.py ──► 3. image_reviewer
                                                        │
4. train_improved.py ◄── setup_improved_training.py ◄──┘
        │
        ▼
5. simple_evaluate.py ──► 6. facebeak.py (process videos)
```

### For Advanced Users (Command Line)
```
extract_training_gui.py → image_reviewer.py → setup_improved_training.py
                                                        │
                                                        ▼
main.py ◄── simple_evaluate.py ◄── train_improved.py ◄──┘
```

### For Researchers (Full Pipeline)
```
Data Extraction → Data Cleaning → Training → Validation → Analysis → Deployment
       │               │             │           │           │           │
       ▼               ▼             ▼           ▼           ▼           ▼
   GUI/CLI         Manual        Automated   Metrics    Clustering   Production
   extraction      review        training    analysis   analysis     processing
```

---

**💡 Pro Tip:** Follow the flowchart from top to bottom, but don't hesitate to loop back to earlier phases if results aren't satisfactory. The pipeline is designed to be iterative and improve over time! 