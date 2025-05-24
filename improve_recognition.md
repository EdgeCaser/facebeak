# Improving Core Recognition in Facebeak

## **Current Issues & Solutions**

### **1. Data Quality & Quantity**
**Problems:**
- Limited training data per crow
- Imbalanced classes (some crows have many images, others few)
- Noisy labels from automatic extraction
- Lack of temporal diversity (all images from same day/conditions)

**Solutions:**
```bash
# 1. Implement data augmentation pipeline
python improve_data.py --crop-dir crow_crops --apply-augmentation --target-samples-per-crow 200

# 2. Add active learning for label cleanup
python image_reviewer.py --mode active_learning --uncertainty-threshold 0.7

# 3. Balance dataset
python balance_dataset.py --min-samples 50 --max-samples 300 --oversample-method triplet
```

### **2. Model Architecture Improvements**
**Current:** Basic ResNet18 → 512D embedding
**Improved:** Multi-scale, attention-based architecture

```python
# Enhanced model with attention and multi-scale features
class EnhancedCrowEmbedder(nn.Module):
    def __init__(self, embedding_dim=512):
        super().__init__()
        # Dual-scale feature extraction
        self.backbone = timm.create_model('efficientnet_b3', pretrained=True, num_classes=0)
        self.head_attention = SpatialAttention()
        self.body_attention = SpatialAttention()
        
        # Multi-scale fusion
        self.fusion = MultiScaleFusion([128, 256, 512])
        self.embedding_head = nn.Sequential(
            nn.Linear(self.backbone.num_features, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(embedding_dim, embedding_dim)
        )
    
    def forward(self, x):
        # Extract multi-scale features
        features = self.backbone.forward_features(x)
        
        # Apply attention to important regions
        head_features = self.head_attention(features)
        body_features = self.body_attention(features)
        
        # Fuse multi-scale information
        fused = self.fusion([head_features, body_features, features])
        
        # Generate embedding
        embedding = self.embedding_head(fused)
        return F.normalize(embedding, p=2, dim=1)
```

### **3. Training Strategy Improvements**

**A. Progressive Training:**
```python
# Stage 1: Coarse recognition (species-level)
python train_progressive.py --stage 1 --epochs 30 --classes species

# Stage 2: Fine-grained recognition (individual-level)  
python train_progressive.py --stage 2 --epochs 50 --classes individual --load-stage1

# Stage 3: Fine-tuning with hard negatives
python train_progressive.py --stage 3 --epochs 20 --hard-mining --adaptive-margin
```

**B. Multi-Task Learning:**
```python
class MultiTaskCrowNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = EnhancedCrowEmbedder()
        
        # Multiple heads for different tasks
        self.identity_head = nn.Linear(512, num_individuals)
        self.pose_head = nn.Linear(512, 8)  # 8 pose classes
        self.age_head = nn.Linear(512, 3)   # juvenile/adult/old
        self.sex_head = nn.Linear(512, 2)   # male/female
        
    def forward(self, x):
        embedding = self.backbone(x)
        return {
            'embedding': embedding,
            'identity': self.identity_head(embedding),
            'pose': self.pose_head(embedding),
            'age': self.age_head(embedding),
            'sex': self.sex_head(embedding)
        }
```

### **4. Advanced Loss Functions**

**A. Replace basic triplet loss:**
```python
# Use the improved triplet loss with adaptive mining
from improved_triplet_loss import ImprovedTripletLoss

criterion = ImprovedTripletLoss(
    margin=1.0,
    mining_type='adaptive',
    margin_schedule=create_margin_schedule('cosine', max_epochs=100)
)
```

**B. Add auxiliary losses:**
```python
class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.triplet_loss = ImprovedTripletLoss(mining_type='adaptive')
        self.center_loss = CenterLoss(num_classes=num_crows, feat_dim=512)
        self.focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
        
    def forward(self, embeddings, labels, predictions=None):
        # Main triplet loss
        triplet_loss, stats = self.triplet_loss(embeddings, labels)
        
        # Center loss for better clustering
        center_loss = self.center_loss(embeddings, labels)
        
        # Classification loss if predictions available
        focal_loss = 0
        if predictions is not None:
            focal_loss = self.focal_loss(predictions, labels)
        
        total_loss = triplet_loss + 0.1 * center_loss + 0.1 * focal_loss
        return total_loss, stats
```

### **5. Data Pipeline Improvements**

**A. Smart crop extraction:**
```python
class SmartCropExtractor:
    def __init__(self):
        self.pose_detector = PoseDetector()
        self.quality_assessor = ImageQualityAssessor()
        
    def extract_crops(self, detections, frame):
        crops = []
        for detection in detections:
            # Assess image quality
            quality_score = self.quality_assessor.score(frame, detection.bbox)
            if quality_score < 0.5:
                continue
                
            # Extract pose-aware crops
            pose = self.pose_detector.detect(frame, detection.bbox)
            crop = self.extract_pose_normalized_crop(frame, detection.bbox, pose)
            
            crops.append({
                'image': crop,
                'quality': quality_score,
                'pose': pose,
                'bbox': detection.bbox
            })
        
        return crops
```

**B. Temporal consistency:**
```python
class TemporalTracker:
    def __init__(self, max_age=30):
        self.tracks = {}
        self.max_age = max_age
        
    def update(self, detections, embeddings):
        # Match detections to existing tracks
        matches = self.match_detections_to_tracks(detections, embeddings)
        
        # Update tracks with temporal smoothing
        for track_id, detection_idx in matches:
            if track_id in self.tracks:
                # Smooth embedding with temporal averaging
                alpha = 0.7  # temporal smoothing factor
                old_embedding = self.tracks[track_id]['embedding']
                new_embedding = embeddings[detection_idx]
                
                smoothed = alpha * old_embedding + (1 - alpha) * new_embedding
                self.tracks[track_id]['embedding'] = F.normalize(smoothed, p=2, dim=0)
                self.tracks[track_id]['age'] = 0
            else:
                # New track
                self.tracks[track_id] = {
                    'embedding': embeddings[detection_idx],
                    'age': 0,
                    'confidence': 1.0
                }
        
        # Age tracks
        for track_id in list(self.tracks.keys()):
            self.tracks[track_id]['age'] += 1
            if self.tracks[track_id]['age'] > self.max_age:
                del self.tracks[track_id]
        
        return self.tracks
```

### **6. Evaluation-Driven Improvements**

**Run comprehensive evaluation:**
```bash
# Evaluate current model
python evaluate_model.py --model-path crow_resnet_triplet.pth --output-dir evaluation_baseline

# Identify weak points
python analyze_failures.py --eval-dir evaluation_baseline --generate-hard-negatives

# Retrain with focus on failure cases
python train_triplet_resnet.py --focus-hard-negatives --curriculum-learning
```

### **7. Real-Time Optimization**

**A. Model distillation:**
```python
class DistilledCrowNet(nn.Module):
    def __init__(self, teacher_model):
        super().__init__()
        self.student = MobileNetV3(num_classes=512)  # Lightweight backbone
        self.teacher = teacher_model
        self.teacher.eval()
        
    def forward(self, x, return_teacher=False):
        student_embedding = self.student(x)
        
        if return_teacher:
            with torch.no_grad():
                teacher_embedding = self.teacher(x)
            return student_embedding, teacher_embedding
        
        return student_embedding

# Knowledge distillation loss
def distillation_loss(student_emb, teacher_emb, temperature=4.0):
    student_soft = F.softmax(student_emb / temperature, dim=1)
    teacher_soft = F.softmax(teacher_emb / temperature, dim=1)
    return F.kl_div(student_soft.log(), teacher_soft, reduction='batchmean')
```

**B. Embedding quantization:**
```python
def quantize_embeddings(embeddings, num_bits=8):
    """Quantize embeddings to reduce memory usage."""
    # Find min/max values
    min_val = embeddings.min()
    max_val = embeddings.max()
    
    # Quantize to num_bits
    scale = (max_val - min_val) / (2**num_bits - 1)
    quantized = torch.round((embeddings - min_val) / scale)
    
    return quantized.to(torch.uint8), scale, min_val
```

## **Implementation Roadmap**

### **Phase 1: Immediate Improvements (1-2 weeks)**
1. Implement improved triplet loss with adaptive mining
2. Add comprehensive evaluation pipeline
3. Clean training data using image reviewer
4. Balance dataset across crow identities

### **Phase 2: Architecture Upgrades (2-4 weeks)**
1. Replace ResNet18 with EfficientNet + attention
2. Add multi-task learning (pose, age, sex prediction)
3. Implement temporal consistency tracking
4. Add center loss and focal loss

### **Phase 3: Advanced Features (4-8 weeks)**
1. Progressive training pipeline
2. Knowledge distillation for mobile deployment
3. Real-time optimization
4. Cross-validation and robustness testing

### **Expected Improvements**
- **Recognition Accuracy:** 60% → 85%+ 
- **False Positive Rate:** 20% → <5%
- **Embedding Quality:** 2x better separability
- **Real-time Performance:** 30% faster inference
- **Robustness:** Better handling of lighting, pose, weather

## **Quick Start Commands**

```bash
# 1. Evaluate current model
python evaluate_model.py --model-path crow_resnet_triplet.pth

# 2. Improve training data
python image_reviewer.py --mode quality_filter --threshold 0.7

# 3. Retrain with improved loss
python train_triplet_resnet.py --loss-type adaptive --mining curriculum

# 4. Test improvements
python evaluate_model.py --model-path training_output/best_model.pth

# 5. Deploy improved model
cp training_output/best_model.pth crow_resnet_triplet.pth
``` 