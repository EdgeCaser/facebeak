#!/usr/bin/env python3
"""
Training Output Cleanup Script

Automatically cleans up training directories by:
1. Extracting key metrics from large JSON files
2. Keeping only essential files (best model, configs, summary)
3. Removing large checkpoint files and redundant data
4. Showing detailed before/after analysis

Usage:
    python cleanup_training_output.py                    # Clean default training_output/
    python cleanup_training_output.py --dir my_training/ # Clean specific directory
    python cleanup_training_output.py --dry-run          # Preview what would be deleted
"""

import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import shutil
from datetime import datetime
import sys

class TrainingCleanup:
    def __init__(self, training_dir="training_output", dry_run=False, interactive=True):
        self.training_dir = Path(training_dir)
        self.dry_run = dry_run
        self.interactive = interactive
        self.cleanup_stats = {
            "files_deleted": 0,
            "space_saved_mb": 0,
            "files_kept": 0,
            "total_files_before": 0,
            "total_size_before_mb": 0,
            "total_size_after_mb": 0
        }
        
    def analyze_directory(self):
        """Analyze the training directory and categorize files."""
        if not self.training_dir.exists():
            print(f"❌ Directory not found: {self.training_dir}")
            return None
            
        files = list(self.training_dir.glob("*"))
        analysis = {
            "essential_files": [],      # Keep these
            "checkpoint_files": [],     # Delete these (large model snapshots)
            "large_metrics": [],        # Extract then delete
            "small_configs": [],        # Keep these
            "visualizations": [],       # Keep these
            "unknown_files": []         # Ask user
        }
        
        total_size = 0
        
        for file_path in files:
            if not file_path.is_file():
                continue
                
            size_mb = file_path.stat().st_size / (1024 * 1024)
            total_size += size_mb
            
            file_info = {
                "path": file_path,
                "name": file_path.name,
                "size_mb": size_mb
            }
            
            # Categorize files
            if file_path.name == "best_model.pth":
                analysis["essential_files"].append(file_info)
            elif file_path.name.startswith("checkpoint_epoch_"):
                analysis["checkpoint_files"].append(file_info)
            elif file_path.name.endswith("_history.json") and size_mb > 100:
                analysis["large_metrics"].append(file_info)
            elif file_path.name.endswith(("config.json", "training_config.json")):
                analysis["small_configs"].append(file_info)
            elif file_path.name.endswith((".png", ".jpg", ".pdf")):
                analysis["visualizations"].append(file_info)
            elif file_path.name.endswith((".json", ".txt", ".log")) and size_mb < 10:
                analysis["small_configs"].append(file_info)
            elif file_path.name.endswith((".pth", ".pt")) and "checkpoint" not in file_path.name:
                analysis["essential_files"].append(file_info)
            else:
                analysis["unknown_files"].append(file_info)
        
        self.cleanup_stats["total_files_before"] = len(files)
        self.cleanup_stats["total_size_before_mb"] = total_size
        
        return analysis
    
    def extract_metrics(self, metrics_files):
        """Extract key metrics from large JSON files."""
        summaries_created = []
        
        for file_info in metrics_files:
            file_path = file_info["path"]
            print(f"📊 Extracting metrics from {file_path.name} ({file_info['size_mb']:.1f} MB)...")
            
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                # Extract key information
                summary = {
                    "extraction_info": {
                        "original_file": file_path.name,
                        "original_size_mb": file_info['size_mb'],
                        "extracted_on": datetime.now().isoformat(),
                        "keys_found": list(data.keys()) if isinstance(data, dict) else "non-dict data"
                    },
                    "training_summary": {},
                    "key_milestones": {}
                }
                
                # Extract training metrics if available
                if isinstance(data, dict):
                    # Common metric names
                    metric_mappings = {
                        "train_loss": ["train_loss", "training_loss", "loss"],
                        "val_loss": ["val_loss", "validation_loss", "val_loss"],
                        "train_acc": ["train_accuracy", "train_acc", "accuracy"],
                        "val_acc": ["val_accuracy", "val_acc", "validation_accuracy"]
                    }
                    
                    for metric_name, possible_keys in metric_mappings.items():
                        for key in possible_keys:
                            if key in data and isinstance(data[key], list) and len(data[key]) > 0:
                                values = data[key]
                                summary["training_summary"][f"{metric_name}_total_epochs"] = len(values)
                                summary["training_summary"][f"{metric_name}_final"] = values[-1]
                                summary["training_summary"][f"{metric_name}_best"] = min(values) if "loss" in metric_name else max(values)
                                
                                # Key milestones (every 10 epochs)
                                milestones = [values[i] for i in range(9, len(values), 10) if i < len(values)]
                                summary["key_milestones"][metric_name] = milestones
                                break
                
                # Save compact summary
                summary_name = file_path.stem + "_summary.json"
                summary_path = file_path.parent / summary_name
                
                if not self.dry_run:
                    with open(summary_path, 'w') as f:
                        json.dump(summary, f, indent=2)
                
                summary_size = len(json.dumps(summary, indent=2)) / 1024  # KB
                summaries_created.append({
                    "original": file_info,
                    "summary_path": summary_path,
                    "summary_size_kb": summary_size,
                    "compression_ratio": (file_info['size_mb'] * 1024) / summary_size
                })
                
                print(f"   ✅ Created {summary_name} ({summary_size:.1f} KB)")
                print(f"   📉 Compression: {file_info['size_mb']:.1f} MB → {summary_size:.1f} KB")
                
            except Exception as e:
                print(f"   ❌ Error extracting from {file_path.name}: {e}")
        
        return summaries_created
    
    def show_analysis(self, analysis):
        """Display detailed analysis of the directory."""
        print(f"\n📂 Analysis of {self.training_dir}")
        print("=" * 60)
        
        categories = [
            ("Essential Files (KEEP)", analysis["essential_files"], "🔥"),
            ("Small Configs (KEEP)", analysis["small_configs"], "📄"),
            ("Visualizations (KEEP)", analysis["visualizations"], "📊"),
            ("Checkpoint Files (DELETE)", analysis["checkpoint_files"], "🗑️"),
            ("Large Metrics (EXTRACT & DELETE)", analysis["large_metrics"], "📈"),
            ("Unknown Files (ASK)", analysis["unknown_files"], "❓")
        ]
        
        total_to_delete_mb = 0
        
        for category_name, files, emoji in categories:
            if not files:
                continue
                
            total_size = sum(f["size_mb"] for f in files)
            print(f"\n{emoji} {category_name}:")
            
            for file_info in files:
                size_str = f"{file_info['size_mb']:.1f} MB" if file_info['size_mb'] >= 1 else f"{file_info['size_mb']*1024:.0f} KB"
                print(f"   📁 {file_info['name']} ({size_str})")
            
            print(f"   📊 Subtotal: {len(files)} files, {total_size:.1f} MB")
            
            if "DELETE" in category_name:
                total_to_delete_mb += total_size
        
        print(f"\n💾 Current total: {len(analysis['essential_files'] + analysis['checkpoint_files'] + analysis['large_metrics'] + analysis['small_configs'] + analysis['visualizations'] + analysis['unknown_files'])} files, {self.cleanup_stats['total_size_before_mb']:.1f} MB")
        print(f"🗑️  Will delete: {total_to_delete_mb:.1f} MB ({total_to_delete_mb/self.cleanup_stats['total_size_before_mb']*100:.1f}% reduction)")
        
        return total_to_delete_mb
    
    def confirm_cleanup(self, total_to_delete_mb):
        """Ask user for confirmation before cleanup."""
        if not self.interactive or self.dry_run:
            return True
            
        print(f"\n⚠️  This will delete {total_to_delete_mb:.1f} MB of files.")
        print("   Large metrics will be summarized before deletion.")
        print("   This action cannot be undone.")
        
        response = input("\n🤔 Proceed with cleanup? [y/N]: ").strip().lower()
        return response in ['y', 'yes']
    
    def perform_cleanup(self, analysis):
        """Perform the actual cleanup operations."""
        print(f"\n🧹 {'DRY RUN - ' if self.dry_run else ''}Starting cleanup...")
        
        # Extract metrics from large files
        if analysis["large_metrics"]:
            print("\n📊 Extracting metrics...")
            summaries = self.extract_metrics(analysis["large_metrics"])
            
            # Add summaries to files to keep
            for summary_info in summaries:
                analysis["small_configs"].append({
                    "path": summary_info["summary_path"],
                    "name": summary_info["summary_path"].name,
                    "size_mb": summary_info["summary_size_kb"] / 1024
                })
        
        # Delete checkpoint files
        files_to_delete = analysis["checkpoint_files"] + analysis["large_metrics"]
        
        for file_info in files_to_delete:
            file_path = file_info["path"]
            print(f"🗑️  {'Would delete' if self.dry_run else 'Deleting'} {file_path.name} ({file_info['size_mb']:.1f} MB)")
            
            if not self.dry_run:
                try:
                    file_path.unlink()
                    self.cleanup_stats["files_deleted"] += 1
                    self.cleanup_stats["space_saved_mb"] += file_info["size_mb"]
                except Exception as e:
                    print(f"   ❌ Error deleting {file_path.name}: {e}")
        
        # Handle unknown files
        if analysis["unknown_files"] and self.interactive and not self.dry_run:
            print(f"\n❓ Found {len(analysis['unknown_files'])} unknown files:")
            for file_info in analysis["unknown_files"]:
                print(f"   📁 {file_info['name']} ({file_info['size_mb']:.1f} MB)")
                response = input(f"     Delete {file_info['name']}? [y/N]: ").strip().lower()
                if response in ['y', 'yes']:
                    try:
                        file_info["path"].unlink()
                        print(f"     🗑️  Deleted {file_info['name']}")
                        self.cleanup_stats["files_deleted"] += 1
                        self.cleanup_stats["space_saved_mb"] += file_info["size_mb"]
                    except Exception as e:
                        print(f"     ❌ Error deleting: {e}")
    
    def show_results(self, analysis):
        """Show cleanup results."""
        files_kept = len(analysis["essential_files"]) + len(analysis["small_configs"]) + len(analysis["visualizations"])
        
        if not self.dry_run:
            # Recalculate final size
            remaining_files = list(self.training_dir.glob("*"))
            final_size = sum(f.stat().st_size for f in remaining_files if f.is_file()) / (1024 * 1024)
            self.cleanup_stats["total_size_after_mb"] = final_size
            self.cleanup_stats["files_kept"] = len(remaining_files)
        else:
            # Estimate for dry run
            self.cleanup_stats["total_size_after_mb"] = self.cleanup_stats["total_size_before_mb"] - self.cleanup_stats["space_saved_mb"]
            self.cleanup_stats["files_kept"] = files_kept
        
        print(f"\n🎉 {'DRY RUN ' if self.dry_run else ''}Cleanup Results:")
        print("=" * 40)
        print(f"📁 Files before: {self.cleanup_stats['total_files_before']}")
        print(f"📁 Files after:  {self.cleanup_stats['files_kept']}")
        print(f"🗑️  Files deleted: {self.cleanup_stats['files_deleted']}")
        print(f"💾 Size before:  {self.cleanup_stats['total_size_before_mb']:.1f} MB")
        print(f"💾 Size after:   {self.cleanup_stats['total_size_after_mb']:.1f} MB")
        print(f"🚀 Space saved:  {self.cleanup_stats['space_saved_mb']:.1f} MB")
        
        if self.cleanup_stats['total_size_before_mb'] > 0:
            reduction_percent = (self.cleanup_stats['space_saved_mb'] / self.cleanup_stats['total_size_before_mb']) * 100
            print(f"📊 Reduction:    {reduction_percent:.1f}%")
    
    def run(self):
        """Run the complete cleanup process."""
        print(f"🧹 Training Output Cleanup Tool")
        print(f"{'🔍 DRY RUN MODE - No files will be deleted' if self.dry_run else '🔥 LIVE MODE - Files will be deleted'}")
        print("=" * 60)
        
        # Analyze directory
        analysis = self.analyze_directory()
        if not analysis:
            return False
        
        # Show analysis
        total_to_delete_mb = self.show_analysis(analysis)
        
        # Confirm cleanup
        if not self.confirm_cleanup(total_to_delete_mb):
            print("❌ Cleanup cancelled by user.")
            return False
        
        # Perform cleanup
        self.perform_cleanup(analysis)
        
        # Show results
        self.show_results(analysis)
        
        if not self.dry_run:
            print(f"\n✅ Cleanup complete! Directory: {self.training_dir}")
        else:
            print(f"\n🔍 Dry run complete. Use --no-dry-run to actually delete files.")
        
        return True

def main():
    parser = argparse.ArgumentParser(description="Clean up training output directories")
    parser.add_argument("--dir", default="training_output", 
                       help="Training directory to clean (default: training_output)")
    parser.add_argument("--dry-run", action="store_true", 
                       help="Preview what would be deleted without actually deleting")
    parser.add_argument("--no-interactive", action="store_true",
                       help="Don't ask for confirmation (use with caution)")
    
    args = parser.parse_args()
    
    cleanup = TrainingCleanup(
        training_dir=args.dir,
        dry_run=args.dry_run,
        interactive=not args.no_interactive
    )
    
    success = cleanup.run()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main() 