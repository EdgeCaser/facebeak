#!/usr/bin/env python3
"""
Quick Start Training Launcher
Automatically starts improved training with optimal parameters for your dataset.
"""

import json
import subprocess
import sys
import os

def main():
    print("🚀 FACEBEAK QUICK START TRAINING")
    print("="*50)
    
    # Check if configuration exists
    if os.path.exists('training_config.json'):
        print("✅ Found training configuration")
        with open('training_config.json', 'r') as f:
            config = json.load(f)
        
        train_config = config['training_config']
        
        # Build optimized command for overnight training
        cmd = [
            sys.executable, 'train_improved.py',
            '--crop-dir', 'crow_crops',
            '--embedding-dim', '256',
            '--epochs', '100',          # More epochs for overnight training
            '--batch-size', '32',       # Utilize your RTX 3080
            '--learning-rate', '0.001',
            '--margin', '1.0',
            '--mining-type', 'adaptive',
            '--output-dir', 'training_output_overnight',
            '--eval-every', '5',
            '--save-every', '10',
            '--early-stopping'
        ]
        
    else:
        print("⚙️ Using default configuration optimized for RTX 3080")
        
        # Default optimized command
        cmd = [
            sys.executable, 'train_improved.py',
            '--crop-dir', 'crow_crops',
            '--embedding-dim', '256',
            '--epochs', '100',
            '--batch-size', '32', 
            '--learning-rate', '0.001',
            '--margin', '1.0',
            '--mining-type', 'adaptive',
            '--output-dir', 'training_output_overnight',
            '--eval-every', '5',
            '--save-every', '10',
            '--early-stopping'
        ]
    
    print("\n🎯 Training Parameters:")
    print("  • Embedding Dimension: 256")
    print("  • Batch Size: 32 (optimized for RTX 3080)")
    print("  • Epochs: 100 (overnight training)")
    print("  • Learning Rate: 0.001")
    print("  • Early Stopping: Enabled")
    print("  • Output: training_output_overnight/")
    
    print(f"\n🚀 Starting training...")
    print(f"Command: {' '.join(cmd)}")
    print("\n" + "="*50)
    
    # Run training
    try:
        subprocess.run(cmd, check=True)
        print("\n✅ Training completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Training failed with exit code: {e.returncode}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n⏹️ Training interrupted by user")
        sys.exit(1)

if __name__ == '__main__':
    main() 