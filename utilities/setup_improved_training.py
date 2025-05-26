#!/usr/bin/env python3
"""
Setup Script for Improved Training
Analyzes your dataset and provides optimal training parameters.
"""

import os
import sys
import json
import argparse
from pathlib import Path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from improved_dataset import DatasetStats

def setup_training_config(crop_dir='crow_crops', output_file='training_config.json'):
    """Setup training configuration based on dataset analysis."""
    
    print("🔍 Analyzing your dataset...")
    print("="*60)
    
    # Analyze dataset
    try:
        recommendations, stats = DatasetStats.recommend_training_params(crop_dir)
        
        # Display dataset statistics
        print("📊 DATASET STATISTICS:")
        print(f"  Total crows: {stats['total_crows']}")
        print(f"  Total images: {stats['total_images']}")
        
        if 'images_per_crow_stats' in stats:
            img_stats = stats['images_per_crow_stats']
            print(f"  Images per crow:")
            print(f"    Min: {img_stats['min']}")
            print(f"    Max: {img_stats['max']}")
            print(f"    Mean: {img_stats['mean']:.1f}")
            print(f"    Median: {img_stats['median']:.1f}")
        
        print()
        print("🎯 RECOMMENDED TRAINING PARAMETERS:")
        for key, value in recommendations.items():
            print(f"  {key}: {value}")
        
        # Create optimal configuration
        config = {
            "dataset_stats": stats,
            "recommended_params": recommendations,
            "training_config": {
                "crop_dir": crop_dir,
                "embedding_dim": recommendations.get('embedding_dim', 256),
                "epochs": recommendations.get('epochs', 50),
                "batch_size": recommendations.get('batch_size', 32),
                "learning_rate": recommendations.get('learning_rate', 0.001),
                "margin": 1.0,
                "mining_type": "adaptive",
                "output_dir": "training_output_improved",
                "eval_every": 5,
                "save_every": 10,
                "weight_decay": 1e-4,
                "alpha": 0.2,
                "beta": 0.02,
                "num_workers": 4,
                "early_stopping": True,
                "plot_every": 10
            }
        }
        
        # Save configuration
        with open(output_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        print()
        print(f"✅ Configuration saved to: {output_file}")
        print()
        print("🚀 READY TO START TRAINING!")
        print("Run the following command:")
        print(f"python train_improved.py --config {output_file}")
        
        # Alternative command line version
        print("\nOr use command line arguments:")
        cmd_args = []
        for key, value in recommendations.items():
            if key.replace('_', '-') in ['embedding-dim', 'epochs', 'batch-size', 'learning-rate']:
                cmd_args.append(f"--{key.replace('_', '-')} {value}")
        
        cmd = f"python train_improved.py {' '.join(cmd_args)}"
        print(f"python train_improved.py {' '.join(cmd_args)}")
        
        return config
        
    except Exception as e:
        print(f"❌ Error analyzing dataset: {e}")
        print("Using default configuration...")
        
        # Default config
        default_config = {
            "training_config": {
                "crop_dir": crop_dir,
                "embedding_dim": 256,
                "epochs": 50,
                "batch_size": 32,
                "learning_rate": 0.001,
                "margin": 1.0,
                "mining_type": "adaptive",
                "output_dir": "training_output_improved",
                "eval_every": 5,
                "save_every": 10
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(default_config, f, indent=2)
        
        return default_config

def create_training_script():
    """Create a simple training launch script."""
    
    script_content = '''#!/usr/bin/env python3
"""
Easy Training Launcher
Automatically uses the best configuration for your dataset.
"""

import json
import subprocess
import sys
import os

def main():
    # Load configuration
    if os.path.exists('training_config.json'):
        with open('training_config.json', 'r') as f:
            config = json.load(f)
        
        train_config = config['training_config']
        
        # Build command
        cmd = [
            sys.executable, 'train_improved.py',
            '--crop-dir', str(train_config['crop_dir']),
            '--embedding-dim', str(train_config['embedding_dim']),
            '--epochs', str(train_config['epochs']),
            '--batch-size', str(train_config['batch_size']),
            '--learning-rate', str(train_config['learning_rate']),
            '--margin', str(train_config['margin']),
            '--mining-type', str(train_config['mining_type']),
            '--output-dir', str(train_config['output_dir']),
            '--eval-every', str(train_config['eval_every']),
            '--save-every', str(train_config['save_every'])
        ]
        
        print("🚀 Starting improved training with optimal parameters...")
        print(f"Command: {' '.join(cmd)}")
        print()
        
        # Run training
        subprocess.run(cmd)
        
    else:
        print("❌ Configuration file not found!")
        print("Please run: python setup_improved_training.py")
        sys.exit(1)

if __name__ == '__main__':
    main()
'''
    
    with open('start_training.py', 'w') as f:
        f.write(script_content)
    
    # Make executable on Unix systems
    if os.name != 'nt':
        os.chmod('start_training.py', 0o755)
    
    print("✅ Created training launcher: start_training.py")

def main():
    parser = argparse.ArgumentParser(description='Setup improved training configuration')
    parser.add_argument('--crop-dir', default='crow_crops', help='Crop directory')
    parser.add_argument('--output', default='training_config.json', help='Output config file')
    parser.add_argument('--create-launcher', action='store_true', help='Create training launcher script')
    
    args = parser.parse_args()
    
    print("🎯 FACEBEAK IMPROVED TRAINING SETUP")
    print("="*60)
    
    # Setup configuration
    config = setup_training_config(args.crop_dir, args.output)
    
    # Create launcher script
    if args.create_launcher:
        print()
        create_training_script()
    
    print()
    print("📋 NEXT STEPS:")
    print("1. Review the recommended parameters above")
    print("2. Run training with: python train_improved.py")
    print("3. Monitor progress in training_output_improved/")
    print("4. Evaluate with: python simple_evaluate.py")
    
    print()
    print("💡 TIPS:")
    print("- Training will automatically save checkpoints")
    print("- Progress plots will be generated every 10 epochs")
    print("- Best model will be saved based on separability metric")
    print("- Use Ctrl+C to stop training and save current progress")

if __name__ == '__main__':
    main() 