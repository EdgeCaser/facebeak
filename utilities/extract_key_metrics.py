#!/usr/bin/env python3
"""
Extract key metrics from large training history file and create a compact summary.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def extract_key_metrics(metrics_file="training_output/metrics_history.json"):
    """Extract and summarize key metrics from the large history file."""
    
    if not Path(metrics_file).exists():
        print(f"Metrics file not found: {metrics_file}")
        return
    
    print(f"Loading metrics from {metrics_file}...")
    print(f"File size: {Path(metrics_file).stat().st_size / (1024*1024):.1f} MB")
    
    try:
        with open(metrics_file, 'r') as f:
            data = json.load(f)
        
        print(f"Loaded data with keys: {list(data.keys())}")
        
        # Extract key information
        summary = {
            "training_summary": {
                "total_epochs": len(data.get("train_loss", [])),
                "final_train_loss": data.get("train_loss", [])[-1] if data.get("train_loss") else None,
                "final_val_loss": data.get("val_loss", [])[-1] if data.get("val_loss") else None,
                "best_val_loss": min(data.get("val_loss", [])) if data.get("val_loss") else None,
                "final_train_acc": data.get("train_accuracy", [])[-1] if data.get("train_accuracy") else None,
                "final_val_acc": data.get("val_accuracy", [])[-1] if data.get("val_accuracy") else None,
                "best_val_acc": max(data.get("val_accuracy", [])) if data.get("val_accuracy") else None,
            },
            "key_milestones": {
                "epochs_10_20_30": {
                    "train_loss": [data.get("train_loss", [])[i] for i in [9, 19, 29] if i < len(data.get("train_loss", []))],
                    "val_loss": [data.get("val_loss", [])[i] for i in [9, 19, 29] if i < len(data.get("val_loss", []))],
                    "train_acc": [data.get("train_accuracy", [])[i] for i in [9, 19, 29] if i < len(data.get("train_accuracy", []))],
                    "val_acc": [data.get("val_accuracy", [])[i] for i in [9, 19, 29] if i < len(data.get("val_accuracy", []))],
                }
            },
            "file_info": {
                "original_size_mb": Path(metrics_file).stat().st_size / (1024*1024),
                "total_data_points": sum(len(v) for v in data.values() if isinstance(v, list)),
                "extracted_on": str(Path().cwd()),
            }
        }
        
        # Save compact summary
        summary_file = "training_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n✅ Extracted key metrics to {summary_file}")
        print(f"Summary file size: {Path(summary_file).stat().st_size / 1024:.1f} KB")
        print(f"Size reduction: {summary['file_info']['original_size_mb']:.1f} MB → {Path(summary_file).stat().st_size / 1024:.1f} KB")
        print(f"Compression ratio: {summary['file_info']['original_size_mb'] * 1024 / (Path(summary_file).stat().st_size / 1024):.0f}:1")
        
        # Create visualization
        if data.get("train_loss") and data.get("val_loss"):
            plt.figure(figsize=(12, 8))
            
            plt.subplot(2, 2, 1)
            plt.plot(data["train_loss"], label="Train Loss")
            plt.plot(data["val_loss"], label="Val Loss") 
            plt.title("Loss Over Time")
            plt.legend()
            plt.grid(True)
            
            if data.get("train_accuracy") and data.get("val_accuracy"):
                plt.subplot(2, 2, 2)
                plt.plot(data["train_accuracy"], label="Train Accuracy")
                plt.plot(data["val_accuracy"], label="Val Accuracy")
                plt.title("Accuracy Over Time")
                plt.legend()
                plt.grid(True)
            
            plt.subplot(2, 2, 3)
            plt.plot(np.gradient(data["train_loss"]), label="Train Loss Gradient")
            plt.title("Loss Change Rate")
            plt.legend()
            plt.grid(True)
            
            plt.subplot(2, 2, 4)
            epochs = len(data["train_loss"])
            plt.bar(["Train Loss", "Val Loss", "Train Acc", "Val Acc"], 
                   [summary["training_summary"]["final_train_loss"],
                    summary["training_summary"]["final_val_loss"], 
                    summary["training_summary"]["final_train_acc"],
                    summary["training_summary"]["final_val_acc"]])
            plt.title("Final Metrics")
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig("training_metrics_summary.png", dpi=150, bbox_inches='tight')
            print(f"📊 Created visualization: training_metrics_summary.png")
        
        print(f"\n📋 Training Summary:")
        print(f"   Total Epochs: {summary['training_summary']['total_epochs']}")
        print(f"   Final Train Loss: {summary['training_summary']['final_train_loss']:.4f}")
        print(f"   Final Val Loss: {summary['training_summary']['final_val_loss']:.4f}")
        print(f"   Best Val Loss: {summary['training_summary']['best_val_loss']:.4f}")
        if summary['training_summary']['final_train_acc']:
            print(f"   Final Train Acc: {summary['training_summary']['final_train_acc']:.4f}")
            print(f"   Final Val Acc: {summary['training_summary']['final_val_acc']:.4f}")
            print(f"   Best Val Acc: {summary['training_summary']['best_val_acc']:.4f}")
        
        return summary
        
    except Exception as e:
        print(f"Error processing metrics file: {e}")
        return None

if __name__ == "__main__":
    print("Extracting Key Training Metrics")
    print("=" * 40)
    extract_key_metrics() 