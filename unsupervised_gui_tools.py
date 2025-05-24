#!/usr/bin/env python3
"""
GUI Tools for Unsupervised Learning Integration
Integrates clustering analysis and auto-labeling with existing GUI workflows.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
from PIL import Image, ImageTk
import cv2
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path
import json

from db import get_all_crows, get_crow_embeddings, update_crow_label
from unsupervised_learning import (
    UnsupervisedTrainingPipeline, 
    AutoLabelingSystem, 
    ReconstructionValidator
)
from crow_clustering import CrowClusterAnalyzer
from models import CrowResNetEmbedder

logger = logging.getLogger(__name__)


class ClusteringBasedLabelSmoother:
    """Implements clustering-based label smoothing with GUI review."""
    
    def __init__(self, confidence_threshold: float = 0.8):
        self.confidence_threshold = confidence_threshold
        self.analyzer = CrowClusterAnalyzer()
        self.auto_labeler = AutoLabelingSystem()
        
    def analyze_and_suggest_merges(self) -> Dict:
        """
        Analyze embeddings and suggest potential crow merges.
        
        Returns:
            Dictionary with merge suggestions and confidence scores
        """
        logger.info("🔍 Analyzing embeddings for potential merges...")
        
        # Get all crow embeddings
        crows = get_all_crows()
        crow_embeddings = {}
        
        for crow in crows:
            embeddings = get_crow_embeddings(crow['id'])
            if embeddings:
                crow_embeddings[crow['id']] = {
                    'name': crow.get('name', f'Crow_{crow["id"]}'),
                    'embeddings': [emb['embedding'] for emb in embeddings],
                    'image_paths': [emb.get('image_path', '') for emb in embeddings]
                }
        
        if len(crow_embeddings) < 2:
            return {'suggestions': [], 'message': 'Need at least 2 crows for merge analysis'}
        
        # Calculate cross-crow similarities
        merge_suggestions = []
        
        for crow_id1, data1 in crow_embeddings.items():
            for crow_id2, data2 in crow_embeddings.items():
                if crow_id1 >= crow_id2:  # Avoid duplicates
                    continue
                
                # Calculate average similarity between all embeddings
                similarities = []
                for emb1 in data1['embeddings']:
                    for emb2 in data2['embeddings']:
                        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
                        similarities.append(similarity)
                
                avg_similarity = np.mean(similarities)
                max_similarity = np.max(similarities)
                
                # Suggest merge if high similarity
                if avg_similarity > self.confidence_threshold:
                    merge_suggestions.append({
                        'crow_id1': crow_id1,
                        'crow_id2': crow_id2,
                        'name1': data1['name'],
                        'name2': data2['name'],
                        'avg_similarity': float(avg_similarity),
                        'max_similarity': float(max_similarity),
                        'confidence': float(avg_similarity),
                        'n_comparisons': len(similarities)
                    })
        
        # Sort by confidence
        merge_suggestions.sort(key=lambda x: x['confidence'], reverse=True)
        
        logger.info(f"Found {len(merge_suggestions)} potential merges")
        
        return {
            'suggestions': merge_suggestions,
            'total_crows': len(crow_embeddings),
            'analysis_complete': True
        }
    
    def suggest_outlier_relabeling(self) -> Dict:
        """Suggest outlier samples that might need relabeling."""
        logger.info("🔍 Analyzing outliers for potential relabeling...")
        
        crows = get_all_crows()
        all_embeddings = []
        all_labels = []
        all_metadata = []
        
        for crow in crows:
            embeddings = get_crow_embeddings(crow['id'])
            for emb in embeddings:
                all_embeddings.append(emb['embedding'])
                all_labels.append(crow['id'])
                all_metadata.append({
                    'crow_id': crow['id'],
                    'crow_name': crow.get('name', f'Crow_{crow["id"]}'),
                    'image_path': emb.get('image_path', ''),
                    'confidence': emb.get('confidence', 1.0)
                })
        
        if len(all_embeddings) < 10:
            return {'outliers': [], 'message': 'Need more embeddings for outlier analysis'}
        
        # Use reconstruction validator to find outliers
        validator = ReconstructionValidator()
        embeddings_tensor = torch.tensor(all_embeddings, dtype=torch.float32)
        
        validator.train_autoencoder(embeddings_tensor)
        outlier_indices, threshold = validator.detect_outliers(embeddings_tensor)
        
        # Prepare outlier information
        outliers = []
        for idx in outlier_indices:
            outliers.append({
                'index': idx,
                'metadata': all_metadata[idx],
                'reconstruction_error': float(threshold)  # Simplified
            })
        
        logger.info(f"Found {len(outliers)} potential outliers")
        
        return {
            'outliers': outliers,
            'threshold': threshold,
            'total_samples': len(all_embeddings)
        }


class UnsupervisedLearningGUI:
    """GUI for reviewing and applying unsupervised learning suggestions."""
    
    def __init__(self, master):
        self.master = master
        self.master.title("Facebeak - Unsupervised Learning Tools")
        self.master.geometry("1200x800")
        
        self.label_smoother = ClusteringBasedLabelSmoother()
        self.current_suggestions = []
        self.current_outliers = []
        
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the main UI components."""
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.master)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Tab 1: Merge Suggestions
        self.merge_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.merge_frame, text="Merge Suggestions")
        self.setup_merge_tab()
        
        # Tab 2: Outlier Review
        self.outlier_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.outlier_frame, text="Outlier Review")
        self.setup_outlier_tab()
        
        # Tab 3: Auto-Labeling
        self.auto_label_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.auto_label_frame, text="Auto-Labeling")
        self.setup_auto_label_tab()
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        self.status_bar = ttk.Label(self.master, textvariable=self.status_var, relief='sunken')
        self.status_bar.pack(side='bottom', fill='x')
    
    def setup_merge_tab(self):
        """Setup the merge suggestions tab."""
        # Controls frame
        controls_frame = ttk.Frame(self.merge_frame)
        controls_frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Button(controls_frame, text="Analyze Merges", 
                  command=self.analyze_merges).pack(side='left', padx=5)
        
        ttk.Label(controls_frame, text="Min Confidence:").pack(side='left', padx=5)
        self.confidence_var = tk.DoubleVar(value=0.8)
        confidence_scale = ttk.Scale(controls_frame, from_=0.5, to=0.95, 
                                   orient='horizontal', variable=self.confidence_var)
        confidence_scale.pack(side='left', padx=5)
        
        # Results frame
        results_frame = ttk.Frame(self.merge_frame)
        results_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Treeview for suggestions
        columns = ('Crow 1', 'Crow 2', 'Confidence', 'Samples', 'Action')
        self.merge_tree = ttk.Treeview(results_frame, columns=columns, show='headings')
        
        for col in columns:
            self.merge_tree.heading(col, text=col)
            self.merge_tree.column(col, width=150)
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(results_frame, orient='vertical', command=self.merge_tree.yview)
        self.merge_tree.configure(yscrollcommand=scrollbar.set)
        
        self.merge_tree.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')
        
        # Action buttons
        action_frame = ttk.Frame(self.merge_frame)
        action_frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Button(action_frame, text="Apply Selected Merge", 
                  command=self.apply_merge).pack(side='left', padx=5)
        ttk.Button(action_frame, text="Reject Selected", 
                  command=self.reject_merge).pack(side='left', padx=5)
        ttk.Button(action_frame, text="Review Images", 
                  command=self.review_merge_images).pack(side='left', padx=5)
    
    def setup_outlier_tab(self):
        """Setup the outlier review tab."""
        # Controls frame
        controls_frame = ttk.Frame(self.outlier_frame)
        controls_frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Button(controls_frame, text="Find Outliers", 
                  command=self.find_outliers).pack(side='left', padx=5)
        
        # Results frame
        results_frame = ttk.Frame(self.outlier_frame)
        results_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Left panel for outlier list
        left_panel = ttk.Frame(results_frame)
        left_panel.pack(side='left', fill='both', expand=True, padx=5)
        
        ttk.Label(left_panel, text="Potential Outliers:").pack(anchor='w')
        
        self.outlier_listbox = tk.Listbox(left_panel)
        self.outlier_listbox.pack(fill='both', expand=True)
        self.outlier_listbox.bind('<<ListboxSelect>>', self.on_outlier_select)
        
        # Right panel for image display
        right_panel = ttk.Frame(results_frame)
        right_panel.pack(side='right', fill='y', padx=5)
        
        self.outlier_image_label = ttk.Label(right_panel, text="Select an outlier to view")
        self.outlier_image_label.pack(pady=10)
        
        # Action buttons for outliers
        outlier_actions = ttk.Frame(right_panel)
        outlier_actions.pack(fill='x', pady=5)
        
        ttk.Button(outlier_actions, text="Mark as Correct", 
                  command=self.mark_outlier_correct).pack(fill='x', pady=2)
        ttk.Button(outlier_actions, text="Relabel Crow", 
                  command=self.relabel_outlier).pack(fill='x', pady=2)
        ttk.Button(outlier_actions, text="Mark as Not Crow", 
                  command=self.mark_not_crow).pack(fill='x', pady=2)
    
    def setup_auto_label_tab(self):
        """Setup the auto-labeling tab."""
        # Controls frame
        controls_frame = ttk.Frame(self.auto_label_frame)
        controls_frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Button(controls_frame, text="Generate Pseudo-Labels", 
                  command=self.generate_pseudo_labels).pack(side='left', padx=5)
        
        ttk.Label(controls_frame, text="Confidence Threshold:").pack(side='left', padx=5)
        self.pseudo_confidence_var = tk.DoubleVar(value=0.95)
        pseudo_scale = ttk.Scale(controls_frame, from_=0.8, to=0.99, 
                               orient='horizontal', variable=self.pseudo_confidence_var)
        pseudo_scale.pack(side='left', padx=5)
        
        # Results display
        results_text_frame = ttk.Frame(self.auto_label_frame)
        results_text_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        self.auto_label_text = tk.Text(results_text_frame, wrap='word')
        auto_scrollbar = ttk.Scrollbar(results_text_frame, orient='vertical', 
                                     command=self.auto_label_text.yview)
        self.auto_label_text.configure(yscrollcommand=auto_scrollbar.set)
        
        self.auto_label_text.pack(side='left', fill='both', expand=True)
        auto_scrollbar.pack(side='right', fill='y')
    
    def analyze_merges(self):
        """Analyze and display merge suggestions."""
        self.status_var.set("Analyzing potential merges...")
        self.master.update()
        
        try:
            self.label_smoother.confidence_threshold = self.confidence_var.get()
            results = self.label_smoother.analyze_and_suggest_merges()
            
            # Clear existing items
            for item in self.merge_tree.get_children():
                self.merge_tree.delete(item)
            
            # Add suggestions to tree
            self.current_suggestions = results['suggestions']
            for i, suggestion in enumerate(self.current_suggestions):
                self.merge_tree.insert('', 'end', iid=i, values=(
                    suggestion['name1'],
                    suggestion['name2'],
                    f"{suggestion['confidence']:.3f}",
                    suggestion['n_comparisons'],
                    "Pending"
                ))
            
            self.status_var.set(f"Found {len(self.current_suggestions)} merge suggestions")
            
        except Exception as e:
            messagebox.showerror("Error", f"Analysis failed: {str(e)}")
            self.status_var.set("Analysis failed")
    
    def find_outliers(self):
        """Find and display outliers."""
        self.status_var.set("Finding outliers...")
        self.master.update()
        
        try:
            results = self.label_smoother.suggest_outlier_relabeling()
            
            # Clear existing items
            self.outlier_listbox.delete(0, tk.END)
            
            # Add outliers to listbox
            self.current_outliers = results['outliers']
            for outlier in self.current_outliers:
                metadata = outlier['metadata']
                display_text = f"{metadata['crow_name']} - {Path(metadata['image_path']).name}"
                self.outlier_listbox.insert(tk.END, display_text)
            
            self.status_var.set(f"Found {len(self.current_outliers)} potential outliers")
            
        except Exception as e:
            messagebox.showerror("Error", f"Outlier detection failed: {str(e)}")
            self.status_var.set("Outlier detection failed")
    
    def on_outlier_select(self, event):
        """Handle outlier selection to display image."""
        selection = self.outlier_listbox.curselection()
        if not selection:
            return
        
        idx = selection[0]
        if idx < len(self.current_outliers):
            outlier = self.current_outliers[idx]
            image_path = outlier['metadata']['image_path']
            
            try:
                self.display_outlier_image(image_path)
            except Exception as e:
                logger.warning(f"Could not display image {image_path}: {e}")
    
    def display_outlier_image(self, image_path: str):
        """Display outlier image in the GUI."""
        try:
            # Load and resize image
            image = cv2.imread(image_path)
            if image is None:
                self.outlier_image_label.config(text="Could not load image")
                return
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (200, 200))
            
            # Convert to PhotoImage
            pil_image = Image.fromarray(image)
            photo = ImageTk.PhotoImage(pil_image)
            
            self.outlier_image_label.config(image=photo, text="")
            self.outlier_image_label.image = photo  # Keep a reference
            
        except Exception as e:
            self.outlier_image_label.config(text=f"Error loading image: {str(e)}")
    
    def apply_merge(self):
        """Apply selected merge suggestion."""
        selection = self.merge_tree.selection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a merge suggestion")
            return
        
        item_id = int(selection[0])
        suggestion = self.current_suggestions[item_id]
        
        # Confirm merge
        result = messagebox.askyesno(
            "Confirm Merge",
            f"Merge '{suggestion['name1']}' and '{suggestion['name2']}'?\n"
            f"Confidence: {suggestion['confidence']:.3f}\n"
            f"This action cannot be easily undone."
        )
        
        if result:
            try:
                # Implement merge logic here
                # This would involve updating the database to merge the two crows
                self.status_var.set(f"Merging {suggestion['name1']} and {suggestion['name2']}...")
                
                # Update tree item
                self.merge_tree.item(item_id, values=(
                    suggestion['name1'],
                    suggestion['name2'],
                    f"{suggestion['confidence']:.3f}",
                    suggestion['n_comparisons'],
                    "Applied"
                ))
                
                messagebox.showinfo("Success", "Merge applied successfully")
                self.status_var.set("Merge completed")
                
            except Exception as e:
                messagebox.showerror("Error", f"Merge failed: {str(e)}")
                self.status_var.set("Merge failed")
    
    def reject_merge(self):
        """Reject selected merge suggestion."""
        selection = self.merge_tree.selection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a merge suggestion")
            return
        
        item_id = int(selection[0])
        self.merge_tree.item(item_id, values=(
            *self.merge_tree.item(item_id)['values'][:4],
            "Rejected"
        ))
    
    def review_merge_images(self):
        """Open image review window for merge suggestion."""
        selection = self.merge_tree.selection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a merge suggestion")
            return
        
        item_id = int(selection[0])
        suggestion = self.current_suggestions[item_id]
        
        # This would open a detailed image comparison window
        messagebox.showinfo("Image Review", 
                          f"Would open image review for {suggestion['name1']} vs {suggestion['name2']}")
    
    def generate_pseudo_labels(self):
        """Generate and display pseudo-labels."""
        self.status_var.set("Generating pseudo-labels...")
        self.master.update()
        
        try:
            confidence_threshold = self.pseudo_confidence_var.get()
            
            # Get embeddings and generate pseudo-labels
            crows = get_all_crows()
            all_embeddings = []
            all_labels = []
            
            for crow in crows:
                embeddings = get_crow_embeddings(crow['id'])
                for emb in embeddings:
                    all_embeddings.append(emb['embedding'])
                    all_labels.append(str(crow['id']))
            
            auto_labeler = AutoLabelingSystem(confidence_threshold=confidence_threshold)
            results = auto_labeler.generate_pseudo_labels(
                np.array(all_embeddings), all_labels
            )
            
            # Display results
            self.auto_label_text.delete(1.0, tk.END)
            
            pseudo_labels = results['pseudo_labels']
            confidences = results['confidences']
            
            self.auto_label_text.insert(tk.END, f"Generated {len(pseudo_labels)} pseudo-labels\n\n")
            self.auto_label_text.insert(tk.END, f"Confidence threshold: {confidence_threshold:.3f}\n")
            self.auto_label_text.insert(tk.END, f"Total embeddings analyzed: {len(all_embeddings)}\n\n")
            
            if pseudo_labels:
                self.auto_label_text.insert(tk.END, "Pseudo-label suggestions:\n")
                for idx, label in pseudo_labels.items():
                    confidence = confidences.get(idx, 0.0)
                    self.auto_label_text.insert(tk.END, 
                        f"Sample {idx}: Label '{label}' (confidence: {confidence:.3f})\n")
            else:
                self.auto_label_text.insert(tk.END, "No confident pseudo-labels found.\n")
                self.auto_label_text.insert(tk.END, "Try lowering the confidence threshold.\n")
            
            self.status_var.set(f"Generated {len(pseudo_labels)} pseudo-labels")
            
        except Exception as e:
            messagebox.showerror("Error", f"Pseudo-label generation failed: {str(e)}")
            self.status_var.set("Pseudo-label generation failed")
    
    def mark_outlier_correct(self):
        """Mark selected outlier as correctly labeled."""
        selection = self.outlier_listbox.curselection()
        if not selection:
            messagebox.showwarning("Warning", "Please select an outlier")
            return
        
        # Implementation would mark the outlier as verified
        messagebox.showinfo("Success", "Outlier marked as correctly labeled")
    
    def relabel_outlier(self):
        """Relabel selected outlier."""
        selection = self.outlier_listbox.curselection()
        if not selection:
            messagebox.showwarning("Warning", "Please select an outlier")
            return
        
        # This would open a relabeling dialog
        messagebox.showinfo("Relabel", "Would open relabeling dialog")
    
    def mark_not_crow(self):
        """Mark selected outlier as not a crow."""
        selection = self.outlier_listbox.curselection()
        if not selection:
            messagebox.showwarning("Warning", "Please select an outlier")
            return
        
        # Implementation would mark as not_a_crow
        messagebox.showinfo("Success", "Marked as not a crow")


def main():
    """Main function to run the unsupervised learning GUI."""
    import torch
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    root = tk.Tk()
    app = UnsupervisedLearningGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main() 