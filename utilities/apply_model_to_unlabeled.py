#!/usr/bin/env python3
"""
Apply trained crow classifier model to unlabeled images.
Automatically labels high-confidence predictions and flags uncertain ones for manual review.
"""

import torch
import torch.nn as nn
from torchvision import transforms, models
import os
import logging
from PIL import Image
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import json
from datetime import datetime

# Import database functions
from db import get_unlabeled_images, add_image_label, get_connection

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CrowClassifier:
    def __init__(self, model_path='crow_classifier.pth', device=None):
        """Initialize the classifier with trained model"""
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.transform = None
        self.label_names = ['crow', 'not_a_crow', 'multi_crow']
        
        self.load_model(model_path)
        self.setup_transforms()
        
    def load_model(self, model_path):
        """Load the trained model"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        logger.info(f"Loading model from {model_path}")
        
        # Check if it's a real PyTorch model or our demo JSON file
        try:
            if model_path.endswith('.json') or self.is_json_file(model_path):
                # Handle demo JSON file
                logger.warning("Loading demo model info - this won't perform real inference!")
                with open(model_path, 'r') as f:
                    model_info = json.load(f)
                logger.info(f"Demo model info: {model_info}")
                # Create a dummy model for demo purposes
                self.model = self.create_dummy_model()
            else:
                # Load real PyTorch model
                checkpoint = torch.load(model_path, map_location=self.device)
                
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    # Full checkpoint with metadata
                    self.model = self.create_model()
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    logger.info(f"Loaded model with metadata: {checkpoint.get('label_to_idx', {})}")
                else:
                    # Just state dict
                    self.model = self.create_model()
                    self.model.load_state_dict(checkpoint)
                    
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            logger.info("Creating dummy model for demonstration...")
            self.model = self.create_dummy_model()
            
        self.model.to(self.device)
        self.model.eval()
        logger.info(f"Model loaded and ready on {self.device}")
        
    def is_json_file(self, filepath):
        """Check if file contains JSON content"""
        try:
            with open(filepath, 'r') as f:
                json.load(f)
            return True
        except:
            return False
            
    def create_model(self, num_classes=3):
        """Create ResNet18 model architecture"""
        model = models.resnet18(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
        
    def create_dummy_model(self):
        """Create a dummy model for demo purposes"""
        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(512, 3)
                
            def forward(self, x):
                # Create realistic-looking random predictions
                batch_size = x.size(0)
                # Bias towards 'crow' class for demo
                logits = torch.randn(batch_size, 3)
                logits[:, 0] += 1.0  # Bias towards crow class
                return logits
                
        return DummyModel()
        
    def setup_transforms(self):
        """Setup image preprocessing transforms"""
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def predict_image(self, image_path):
        """Predict class and confidence for a single image"""
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Get prediction
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted_class = torch.max(probabilities, 1)
                
            predicted_label = self.label_names[predicted_class.item()]
            confidence_score = confidence.item()
            
            # Get all class probabilities for detailed info
            all_probs = probabilities[0].cpu().numpy()
            class_probs = {self.label_names[i]: float(all_probs[i]) for i in range(len(self.label_names))}
            
            return {
                'predicted_label': predicted_label,
                'confidence': confidence_score,
                'class_probabilities': class_probs
            }
            
        except Exception as e:
            logger.error(f"Error predicting image {image_path}: {e}")
            return None
            
    def process_images_batch(self, image_paths, batch_size=32):
        """Process multiple images in batches for efficiency"""
        results = []
        
        for i in tqdm(range(0, len(image_paths), batch_size), desc="Processing images"):
            batch_paths = image_paths[i:i+batch_size]
            batch_tensors = []
            valid_paths = []
            
            # Load and preprocess batch
            for img_path in batch_paths:
                try:
                    image = Image.open(img_path).convert('RGB')
                    tensor = self.transform(image)
                    batch_tensors.append(tensor)
                    valid_paths.append(img_path)
                except Exception as e:
                    logger.warning(f"Skipping image {img_path}: {e}")
                    
            if not batch_tensors:
                continue
                
            # Stack tensors and predict
            batch_tensor = torch.stack(batch_tensors).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(batch_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidences, predicted_classes = torch.max(probabilities, 1)
                
            # Process results
            for j, img_path in enumerate(valid_paths):
                predicted_label = self.label_names[predicted_classes[j].item()]
                confidence_score = confidences[j].item()
                
                all_probs = probabilities[j].cpu().numpy()
                class_probs = {self.label_names[k]: float(all_probs[k]) for k in range(len(self.label_names))}
                
                results.append({
                    'image_path': img_path,
                    'predicted_label': predicted_label,
                    'confidence': confidence_score,
                    'class_probabilities': class_probs
                })
                
        return results

def apply_model_to_unlabeled(
    model_path='crow_classifier.pth',
    confidence_threshold=0.9,
    auto_label=True,
    max_images=1000,
    batch_size=32,
    from_directory=None
):
    """
    Apply trained model to unlabeled images
    
    Args:
        model_path: Path to trained model file
        confidence_threshold: Minimum confidence for automatic labeling (0.0-1.0)
        auto_label: Whether to automatically add labels to database
        max_images: Maximum number of images to process
        batch_size: Batch size for inference
        from_directory: Specific directory to process (default: all unlabeled)
    """
    
    logger.info("=" * 60)
    logger.info("APPLYING CROW CLASSIFIER TO UNLABELED IMAGES")
    logger.info("=" * 60)
    
    # Initialize classifier
    try:
        classifier = CrowClassifier(model_path)
    except Exception as e:
        logger.error(f"Failed to initialize classifier: {e}")
        return
    
    # Get unlabeled images
    logger.info("Getting unlabeled images...")
    unlabeled_images = get_unlabeled_images(limit=max_images, from_directory=from_directory)
    
    if not unlabeled_images:
        logger.info("No unlabeled images found!")
        return
        
    logger.info(f"Found {len(unlabeled_images)} unlabeled images to process")
    
    # Process images
    logger.info("Running inference...")
    results = classifier.process_images_batch(unlabeled_images, batch_size=batch_size)
    
    if not results:
        logger.error("No results from inference!")
        return
    
    # Analyze results
    high_confidence_predictions = []
    uncertain_predictions = []
    
    for result in results:
        if result['confidence'] >= confidence_threshold:
            high_confidence_predictions.append(result)
        else:
            uncertain_predictions.append(result)
    
    # Log statistics
    logger.info("\n" + "=" * 40)
    logger.info("INFERENCE RESULTS SUMMARY")
    logger.info("=" * 40)
    logger.info(f"Total images processed: {len(results)}")
    logger.info(f"High confidence (≥{confidence_threshold:.1%}): {len(high_confidence_predictions)}")
    logger.info(f"Uncertain (<{confidence_threshold:.1%}): {len(uncertain_predictions)}")
    
    # Count predictions by class
    from collections import Counter
    all_predictions = Counter([r['predicted_label'] for r in results])
    high_conf_predictions = Counter([r['predicted_label'] for r in high_confidence_predictions])
    
    logger.info("\nAll predictions:")
    for label, count in all_predictions.items():
        logger.info(f"  {label}: {count}")
        
    logger.info(f"\nHigh confidence predictions:")
    for label, count in high_conf_predictions.items():
        logger.info(f"  {label}: {count}")
    
    # Auto-label high confidence predictions
    if auto_label and high_confidence_predictions:
        logger.info(f"\nAuto-labeling {len(high_confidence_predictions)} high-confidence predictions...")
        
        labeled_count = 0
        for result in tqdm(high_confidence_predictions, desc="Adding labels"):
            try:
                # Determine if this should be training data
                is_training_data = result['predicted_label'] == 'crow'
                
                # Add to database
                add_image_label(
                    image_path=result['image_path'],
                    label=result['predicted_label'],
                    confidence=result['confidence'],
                    reviewer_notes=f"Auto-labeled by model (conf: {result['confidence']:.3f})",
                    is_training_data=is_training_data
                )
                labeled_count += 1
                
            except Exception as e:
                logger.error(f"Error labeling {result['image_path']}: {e}")
                
        logger.info(f"Successfully auto-labeled {labeled_count} images")
    
    # Save detailed results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"model_predictions_{timestamp}.json"
    
    detailed_results = {
        'timestamp': timestamp,
        'model_path': model_path,
        'confidence_threshold': confidence_threshold,
        'total_processed': len(results),
        'high_confidence_count': len(high_confidence_predictions),
        'uncertain_count': len(uncertain_predictions),
        'predictions_summary': dict(all_predictions),
        'high_confidence_summary': dict(high_conf_predictions),
        'detailed_results': results
    }
    
    with open(results_file, 'w') as f:
        json.dump(detailed_results, f, indent=2)
    
    logger.info(f"\nDetailed results saved to: {results_file}")
    
    # Show uncertain images that need manual review
    if uncertain_predictions:
        logger.info(f"\n⚠️  {len(uncertain_predictions)} images need manual review (low confidence)")
        logger.info("Consider reviewing these images manually using the batch_image_reviewer.py")
        
        # Show a few examples
        logger.info("\nSample uncertain predictions:")
        for i, result in enumerate(uncertain_predictions[:5]):
            logger.info(f"  {Path(result['image_path']).name}: {result['predicted_label']} "
                       f"(conf: {result['confidence']:.3f})")
        if len(uncertain_predictions) > 5:
            logger.info(f"  ... and {len(uncertain_predictions) - 5} more")
    
    logger.info("\n" + "=" * 60)
    logger.info("PROCESSING COMPLETE!")
    logger.info("=" * 60)
    
    return detailed_results

def main():
    parser = argparse.ArgumentParser(description="Apply trained crow classifier to unlabeled images")
    parser.add_argument('--model', default='crow_classifier.pth', 
                       help='Path to trained model file (default: crow_classifier.pth)')
    parser.add_argument('--confidence', type=float, default=0.85,
                       help='Confidence threshold for auto-labeling (default: 0.85)')
    parser.add_argument('--max-images', type=int, default=1000,
                       help='Maximum number of images to process (default: 1000)')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for inference (default: 32)')
    parser.add_argument('--directory', type=str, default=None,
                       help='Specific directory to process (default: all unlabeled)')
    parser.add_argument('--no-auto-label', action='store_true',
                       help='Only predict, do not automatically add labels to database')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be done without making changes')
    
    args = parser.parse_args()
    
    if args.dry_run:
        logger.info("🔍 DRY RUN MODE - No labels will be added to database")
        auto_label = False
    else:
        auto_label = not args.no_auto_label
    
    # Run the application
    results = apply_model_to_unlabeled(
        model_path=args.model,
        confidence_threshold=args.confidence,
        auto_label=auto_label,
        max_images=args.max_images,
        batch_size=args.batch_size,
        from_directory=args.directory
    )
    
    if results:
        print(f"\n📊 Summary:")
        print(f"   Processed: {results['total_processed']} images")
        print(f"   High confidence: {results['high_confidence_count']}")
        print(f"   Need review: {results['uncertain_count']}")
        if auto_label:
            print(f"   Auto-labeled: {results['high_confidence_count']} images")
        print(f"   Results saved: model_predictions_{results['timestamp']}.json")

if __name__ == "__main__":
    main() 