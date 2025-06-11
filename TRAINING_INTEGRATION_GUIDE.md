# Training Integration Guide: Non-Crow Images

## Overview

This guide explains how to integrate your **2,097 non-crow images** into your crow training pipeline. The images have been properly processed to avoid letterboxing and use actual bounding boxes where available.

## Dataset Summary

### ✅ Processed Non-Crow Images: 2,097 total

**CUB Bird Images: 897**
- **117 images** with bounding box annotations (properly cropped using bboxes)
- **780 images** without bboxes (center-cropped to square)
- **Species:** American Goldfinch, Cardinal, Blue Jay, Mallard, Chipping Sparrow, etc.

**COCO Object Images: 1,200**
- **1,200 images** with bounding box annotations (properly cropped using bboxes)
- **Categories:** car, dog, cat, person, bicycle, bus, chair, couch, tv, laptop
- **Processing:** Uses COCO API to extract actual object instances with precise bounding boxes

### ✅ Processing Quality
- **No letterboxing** - all images are properly cropped, not padded with black
- **512x512 resolution** - consistent with your training pipeline
- **Aspect ratio preserved** - no distortion of objects
- **Precise object crops** - COCO images use actual object bounding boxes

## Integration Methods

### Method 1: Crow Classifier Training (Recommended)

Use the updated `utilities/train_crow_classifier.py` to train a binary classifier that can distinguish crows from non-crows.

#### Step 1: Run the Crow Classifier Training

```bash
python utilities/train_crow_classifier.py
```

**What this does:**
- Loads your labeled crow images from the database
- Adds all 1,497 non-crow images with 'not_a_crow' labels
- Trains a ResNet18 model to classify: crow vs not_a_crow vs multi_crow
- Uses 512x512 input size with proper aspect ratio transforms
- Saves the trained model as `crow_classifier.pth`

#### Step 2: Use the Classifier to Filter Your Dataset

```python
# Load the trained classifier
import torch
from utilities.train_crow_classifier import create_model, get_transforms

model_info = torch.load('crow_classifier.pth')
model = create_model(num_classes=3)
model.load_state_dict(model_info['model_state_dict'])
model.eval()

# Use to filter images
def is_crow(image_path, confidence_threshold=0.8):
    transform = get_transforms()[1]  # Use validation transform
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.softmax(outputs, dim=1)
        crow_prob = probabilities[0][0].item()  # Class 0 = crow
        
    return crow_prob > confidence_threshold
```

### Method 2: Triplet Training with Hard Negatives

Use the non-crow images as hard negative examples in your triplet training.

#### Step 1: Update Your Triplet Dataset

Modify your `improved_dataset.py` to include non-crow images:

```python
class ImprovedCrowTripletDataset(Dataset):
    def __init__(self, base_dir, split='train', transform_mode='standard',
                 include_non_crow=True, non_crow_dir="not_crow_samples_cropped_512"):
        # ... existing initialization ...
        
        if include_non_crow:
            self._load_non_crow_samples(non_crow_dir)
    
    def _load_non_crow_samples(self, non_crow_dir):
        """Load non-crow images as negative examples"""
        non_crow_dir = Path(non_crow_dir)
        if not non_crow_dir.exists():
            logger.warning(f"Non-crow directory not found: {non_crow_dir}")
            return
            
        non_crow_images = []
        for img_path in non_crow_dir.glob("*.jpg"):
            non_crow_images.append((img_path, "non_crow"))
        
        # Add to samples
        self.samples.extend(non_crow_images)
        logger.info(f"Added {len(non_crow_images)} non-crow images as negative examples")
    
    def _get_negative_sample(self, anchor_crow_id):
        """Get a negative sample (different crow or non-crow)"""
        if anchor_crow_id == "non_crow":
            # If anchor is non-crow, get a crow as negative
            crow_samples = [(path, crow_id) for path, crow_id in self.samples 
                           if crow_id != "non_crow"]
            if crow_samples:
                return random.choice(crow_samples)
        
        # Get non-crow or different crow
        possible_negatives = [(path, crow_id) for path, crow_id in self.samples 
                             if crow_id != anchor_crow_id]
        
        if not possible_negatives:
            return self.samples[0]  # Fallback
        
        return random.choice(possible_negatives)
```

#### Step 2: Train with Hard Negatives

```python
# Initialize dataset with non-crow images
dataset = ImprovedCrowTripletDataset(
    base_dir="crow_crops",
    split='train',
    include_non_crow=True,
    non_crow_dir="not_crow_samples_cropped_512"
)

# Train your triplet model
# The model will learn to distinguish crows from non-crows
```

### Method 3: Evaluation and Testing

Use the non-crow images to evaluate your model's ability to distinguish crows from other objects.

#### Step 1: Create Evaluation Script

```python
def evaluate_crow_vs_non_crow(model, crow_dir, non_crow_dir):
    """Evaluate model's ability to distinguish crows from non-crows"""
    
    # Load crow embeddings
    crow_embeddings = []
    for img_path in Path(crow_dir).glob("*.jpg"):
        embedding = model.extract_embedding(img_path)
        crow_embeddings.append(embedding)
    
    # Load non-crow embeddings
    non_crow_embeddings = []
    for img_path in Path(non_crow_dir).glob("*.jpg"):
        embedding = model.extract_embedding(img_path)
        non_crow_embeddings.append(embedding)
    
    # Calculate similarities
    crow_similarities = compute_similarities(crow_embeddings)
    non_crow_similarities = compute_similarities(non_crow_embeddings)
    
    # Evaluate separation
    print(f"Crow similarity: {np.mean(crow_similarities):.3f}")
    print(f"Non-crow similarity: {np.mean(non_crow_similarities):.3f}")
    print(f"Separation: {np.mean(crow_similarities) - np.mean(non_crow_similarities):.3f}")
```

## Training Pipeline Integration

### 1. Data Loading Verification

Test that your data loading works correctly:

```bash
python test_data_loading.py
```

Expected output:
```
Testing non-crow image loading...
Added 2097 non-crow images from not_crow_samples_cropped_512
Total dataset: 2097 images
Label distribution: {'not_a_crow': 2097}
```

### 2. Image Quality Verification

Check that images are properly processed:

```bash
python check_images.py
```

Expected output:
```
Letterboxed: False  # Should be False for all images
```

### 3. Training Commands

#### Crow Classifier Training:
```bash
python utilities/train_crow_classifier.py
```

#### Triplet Training with Non-Crow:
```python
# In your training script
dataset = ImprovedCrowTripletDataset(
    base_dir="crow_crops",
    include_non_crow=True
)
```

## Benefits of This Integration

### ✅ Improved Model Performance
- **Better crow detection** - model learns what crows look like vs other birds/objects
- **Reduced false positives** - fewer non-crows mistaken for crows
- **Harder negatives** - more challenging training examples

### ✅ Realistic Training Data
- **Diverse negative examples** - birds, cars, people, furniture, etc.
- **Proper aspect ratios** - no distortion artifacts
- **Consistent processing** - all images processed the same way

### ✅ Evaluation Capabilities
- **Crow vs non-crow metrics** - measure model's discrimination ability
- **Hard negative mining** - use non-crow images as challenging negatives
- **Quality filtering** - use classifier to clean your dataset

## Next Steps

1. **Train the crow classifier** to get a baseline model
2. **Integrate non-crow images** into your triplet training
3. **Evaluate performance** using crow vs non-crow metrics
4. **Iterate and improve** based on results

## Troubleshooting

### Issue: "No module named 'db'"
**Solution:** The training script requires your database module. Either:
- Install your database dependencies
- Or modify the script to work without database labels

### Issue: Memory problems with large dataset
**Solution:** 
- Reduce batch size in training
- Use data loading with `num_workers=0`
- Process images in smaller batches

### Issue: Poor training performance
**Solution:**
- Check image quality with `check_images.py`
- Verify data loading with `test_data_loading.py`
- Adjust learning rate and training parameters
- Consider class balancing if dataset is imbalanced

## File Structure

```
facebeak/
├── not_crow_samples_cropped_512/     # 2,097 processed non-crow images
│   ├── cub_*.jpg                     # 897 CUB bird images
│   └── coco_*.jpg                    # 1,200 COCO object images
├── utilities/
│   ├── train_crow_classifier.py      # Updated for non-crow integration
│   └── ...
├── crop_not_crow_samples.py          # Fixed cropping script
├── check_images.py                   # Image quality verification
├── test_data_loading.py              # Data loading test
└── TRAINING_INTEGRATION_GUIDE.md     # This guide
```

Your non-crow dataset is now ready for training! 🎯 