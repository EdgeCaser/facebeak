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
Input Videos ──► Detection Models ──► Crow Images ──► Audio Segments
     │                  │                  │              │
     │            ┌─────────────┐          │              │
     │            │ Faster R-CNN│          │              │
     │            │   YOLOv8    │          │              │
     │            └─────────────┘          │              │
     │                                     │              │
     └─────────────────────────────────────┼──────────────┘
                                           │
                                           ▼
                                    crow_crops/
                                    ├── crow_0001/
                                    ├── crow_0002/
                                    └── audio/
```

### Phase 2: Data Cleaning (15-30 min per 1000 images)
```
Raw Crow Images ──► Manual Review ──► Clean Dataset
       │                  │                │
   ┌───────────┐    ┌─────────────┐   ┌─────────────┐
   │ All crops │    │1=Crow       │   │ Verified    │
   │ (mixed    │───►│2=Not crow   │──►│ crow images │
   │ quality)  │    │3=Unsure     │   │ only        │
   └───────────┘    │4=Multiple   │   └─────────────┘
                    └─────────────┘
```

### Phase 3: Training Setup (5 min)
```
Clean Dataset ──► Analysis ──► Configuration ──► Ready to Train
      │              │             │                  │
  ┌─────────┐   ┌─────────┐   ┌─────────────┐   ┌─────────────┐
  │ Count   │   │Optimal  │   │training_    │   │ Estimated   │
  │ crows   │──►│batch    │──►│config.json  │──►│ time &      │
  │ images  │   │size etc │   │             │   │ resources   │
  └─────────┘   └─────────┘   └─────────────┘   └─────────────┘
```

### Phase 4: Model Training (8-72 hours)
```
Training Data ──► Neural Network ──► Trained Model
      │                  │                │
  ┌─────────┐      ┌─────────────┐   ┌─────────────┐
  │ Triplet │      │ ResNet-18   │   │ 512D        │
  │ samples │─────►│ + Triplet   │──►│ embeddings  │
  │         │      │ Loss        │   │ model       │
  └─────────┘      └─────────────┘   └─────────────┘
                         │
                         ▼
                   ┌─────────────┐
                   │ Checkpoints │
                   │ every 10    │
                   │ epochs      │
                   └─────────────┘
```

### Phase 5: Validation (5-10 min)
```
Trained Model ──► Test Data ──► Performance Metrics
      │              │               │
  ┌─────────┐   ┌─────────┐    ┌─────────────┐
  │ .pth    │   │ Held-out│    │Separability │
  │ file    │──►│ crow    │───►│Same/Diff    │
  │         │   │ images  │    │similarity   │
  └─────────┘   └─────────┘    └─────────────┘
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