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

from db import get_all_crows, get_crow_embeddings, update_crow_name, reassign_crow_embeddings, get_connection
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

    def perform_merge_operation(self, crow_id_from: int, crow_id_to: int) -> Tuple[bool, str]:
        """
        Perform the merge operation: reassign embeddings from crow_id_from to crow_id_to,
        and then delete crow_id_from.

        Args:
            crow_id_from: The ID of the crow to merge from.
            crow_id_to: The ID of the crow to merge into.

        Returns:
            A tuple (success: bool, message: str).
        """
        logger.info(f"Attempting to merge crow {crow_id_from} into crow {crow_id_to}")
        try:
            # Step 1: Reassign all embeddings from crow_id_from to crow_id_to
            # The reassign_crow_embeddings function handles updates to sighting counts and timestamps.
            moved_count = reassign_crow_embeddings(from_crow_id=crow_id_from, to_crow_id=crow_id_to)
            logger.info(f"Reassigned {moved_count} embeddings from crow {crow_id_from} to {crow_id_to}")

            # Step 2: Delete the original crow (crow_id_from)
            # Since db.py doesn't have a dedicated delete_crow, we'll do it here.
            # Note: This assumes that related data in other tables (e.g., behavioral_markers
            # linked to embeddings of crow_id_from) are either handled by ON DELETE CASCADE,
            # or are implicitly handled because all embeddings were moved.
            # The reassign_crow_embeddings function sets total_sightings of from_crow_id to 0
            # if all embeddings are moved, but doesn't delete the crow row.
            
            conn = None
            try:
                conn = get_connection()
                cursor = conn.cursor()
                # Ensure there are no remaining embeddings for crow_id_from, just in case.
                # This should ideally be 0 after reassign_crow_embeddings.
                cursor.execute("SELECT COUNT(*) FROM crow_embeddings WHERE crow_id = ?", (crow_id_from,))
                remaining_embeddings = cursor.fetchone()[0]
                if remaining_embeddings > 0:
                    # This case should ideally not happen if reassign_crow_embeddings worked as expected for all embeddings.
                    # However, if reassign_crow_embeddings was called with specific embedding_ids (not the case here),
                    # or if there's an issue, this is a safeguard.
                    logger.warning(f"Crow {crow_id_from} still has {remaining_embeddings} embeddings. Deleting them now.")
                    cursor.execute("DELETE FROM crow_embeddings WHERE crow_id = ?", (crow_id_from,))
                
                # Now, delete the crow from the crows table
                cursor.execute("DELETE FROM crows WHERE id = ?", (crow_id_from,))
                conn.commit()
                logger.info(f"Successfully deleted crow {crow_id_from} from crows table.")
            except Exception as e_db_delete:
                if conn:
                    conn.rollback()
                logger.error(f"Database error while deleting crow {crow_id_from}: {e_db_delete}")
                return False, f"Failed to delete crow {crow_id_from}: {e_db_delete}"
            finally:
                if conn:
                    conn.close()
            
            return True, f"Successfully merged crow {crow_id_from} into {crow_id_to}."

        except Exception as e:
            logger.error(f"Error during merge operation for {crow_id_from} -> {crow_id_to}: {e}")
            return False, f"Merge failed: {str(e)}"
        
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
        
        output = {
            'suggestions': merge_suggestions,
            'total_crows': len(crow_embeddings),
            'analysis_complete': True
        }
        del crow_embeddings # Free up memory
        return output
    
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
        try:
            del all_embeddings # Free memory from the list of embeddings
            logger.debug("Deleted all_embeddings list after converting to tensor in suggest_outlier_relabeling.")
        except NameError:
            pass # Should exist
        
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
        
        output = {
            'outliers': outliers,
            'threshold': threshold,
            'total_samples': len(all_embeddings)
        }
        # Clean up large data structures
        try:
            # all_embeddings should have been deleted earlier
            del embeddings_tensor
            del all_metadata
            logger.debug("Cleaned up embedding tensor and metadata in suggest_outlier_relabeling")
        except NameError:
            logger.debug("Some embedding structures (tensor or metadata) were not defined in suggest_outlier_relabeling.")
            pass
        return output


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
            self.status_var.set(f"Merging {suggestion['name1']} into {suggestion['name2']}...")
            self.master.update() # Ensure status bar updates

            try:
                crow_id_from = suggestion['crow_id1']
                crow_id_to = suggestion['crow_id2']
                
                # Ensure crow_id_from is the one with fewer sightings or older if equal,
                # to preserve the more established crow ID if possible.
                # However, the suggestion already has name1 and name2, id1 and id2.
                # For simplicity, we'll assume the user is prompted to confirm which ID to keep,
                # or the suggestion implies id1 is merged into id2.
                # The current suggestion['crow_id1'] will be merged into suggestion['crow_id2'].

                success, message = self.label_smoother.perform_merge_operation(crow_id_from, crow_id_to)

                if success:
                    self.merge_tree.item(item_id, values=(
                        suggestion['name1'], # Or the name of the merged crow
                        suggestion['name2'], # Or the name of the merged crow
                        f"{suggestion['confidence']:.3f}",
                        suggestion['n_comparisons'], # This might change or be irrelevant
                        "Applied"
                    ))
                    messagebox.showinfo("Success", message)
                    self.status_var.set("Merge completed")
                    # Optional: Refresh merge suggestions list or remove the applied one
                    self.analyze_merges()
                else:
                    messagebox.showerror("Error", f"Merge failed: {message}")
                    self.status_var.set("Merge failed")
            except Exception as e:
                # This specific exception catch might be redundant if perform_merge_operation handles all its exceptions
                # and returns False, message. But kept for safety.
                logger.error(f"GUI error during merge: {str(e)}")
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
        
        # Open image review window
        self.open_image_review_window(suggestion)
    
    def open_image_review_window(self, suggestion):
        """Open a window to review images for merge decision."""
        review_window = tk.Toplevel(self.master)
        review_window.title(f"Image Review: {suggestion['name1']} vs {suggestion['name2']}")
        review_window.geometry("800x600")
        
        # Header with suggestion info
        header_frame = ttk.Frame(review_window)
        header_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Label(header_frame, text=f"Comparing: {suggestion['name1']} (ID: {suggestion['crow_id1']}) vs {suggestion['name2']} (ID: {suggestion['crow_id2']})", 
                 font=('Arial', 12, 'bold')).pack()
        ttk.Label(header_frame, text=f"Confidence: {suggestion['confidence']:.3f} | Comparisons: {suggestion['n_comparisons']}").pack()
        
        # Main content frame
        content_frame = ttk.Frame(review_window)
        content_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Left side - Crow 1 images
        left_frame = ttk.LabelFrame(content_frame, text=f"{suggestion['name1']} (ID: {suggestion['crow_id1']})")
        left_frame.pack(side='left', fill='both', expand=True, padx=5)
        
        # Right side - Crow 2 images
        right_frame = ttk.LabelFrame(content_frame, text=f"{suggestion['name2']} (ID: {suggestion['crow_id2']})")
        right_frame.pack(side='right', fill='both', expand=True, padx=5)
        
        # Load and display images
        self.load_crow_images_for_review(left_frame, suggestion['crow_id1'])
        self.load_crow_images_for_review(right_frame, suggestion['crow_id2'])
        
        # Action buttons
        button_frame = ttk.Frame(review_window)
        button_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Button(button_frame, text="Merge These Crows", 
                  command=lambda: self.confirm_merge_from_review(review_window, suggestion)).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Keep Separate", 
                  command=lambda: self.reject_merge_from_review(review_window, suggestion)).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Close", 
                  command=review_window.destroy).pack(side='right', padx=5)
    
    def load_crow_images_for_review(self, parent_frame, crow_id):
        """Load and display images for a specific crow in the review window."""
        try:
            embeddings = get_crow_embeddings(crow_id)
            
            if not embeddings:
                ttk.Label(parent_frame, text="No images found").pack(pady=20)
                return
            
            # Create scrollable frame for images
            canvas = tk.Canvas(parent_frame, height=400)
            scrollbar = ttk.Scrollbar(parent_frame, orient="vertical", command=canvas.yview)
            scrollable_frame = ttk.Frame(canvas)
            
            scrollable_frame.bind(
                "<Configure>",
                lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
            )
            
            canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
            canvas.configure(yscrollcommand=scrollbar.set)
            
            # Load up to 10 images to avoid overwhelming the UI
            max_images = min(10, len(embeddings))
            images_per_row = 2
            
            for i, emb in enumerate(embeddings[:max_images]):
                image_path = emb.get('image_path', '')
                if not image_path or not Path(image_path).exists():
                    continue
                
                try:
                    # Load and resize image
                    image = cv2.imread(image_path)
                    if image is None:
                        continue
                    
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    image = cv2.resize(image, (150, 150))
                    
                    # Convert to PhotoImage
                    pil_image = Image.fromarray(image)
                    photo = ImageTk.PhotoImage(pil_image)
                    
                    # Calculate grid position
                    row = i // images_per_row
                    col = i % images_per_row
                    
                    # Create frame for image and info
                    img_frame = ttk.Frame(scrollable_frame)
                    img_frame.grid(row=row, column=col, padx=5, pady=5, sticky='nsew')
                    
                    # Image label
                    img_label = ttk.Label(img_frame, image=photo)
                    img_label.image = photo  # Keep reference
                    img_label.pack()
                    
                    # Image info
                    info_text = f"Conf: {emb.get('confidence', 'N/A'):.3f}\n{Path(image_path).name}"
                    ttk.Label(img_frame, text=info_text, font=('Arial', 8)).pack()
                    
                except Exception as e:
                    logger.warning(f"Could not load image {image_path}: {e}")
                    continue
            
            canvas.pack(side="left", fill="both", expand=True)
            scrollbar.pack(side="right", fill="y")
            
            # Show total count
            ttk.Label(parent_frame, text=f"Showing {min(max_images, len(embeddings))} of {len(embeddings)} images").pack(pady=5)
            
        except Exception as e:
            logger.error(f"Error loading images for crow {crow_id}: {e}")
            ttk.Label(parent_frame, text=f"Error loading images: {str(e)}").pack(pady=20)
    
    def confirm_merge_from_review(self, review_window, suggestion):
        """Confirm merge after image review."""
        review_window.destroy()
        
        # Find the suggestion in the tree and apply merge
        for item in self.merge_tree.get_children():
            values = self.merge_tree.item(item)['values']
            if (values[0] == suggestion['name1'] and 
                values[1] == suggestion['name2'] and 
                float(values[2]) == suggestion['confidence']):
                
                self.merge_tree.selection_set(item)
                self.apply_merge()
                break
    
    def reject_merge_from_review(self, review_window, suggestion):
        """Reject merge after image review."""
        review_window.destroy()
        
        # Find the suggestion in the tree and reject it
        for item in self.merge_tree.get_children():
            values = self.merge_tree.item(item)['values']
            if (values[0] == suggestion['name1'] and 
                values[1] == suggestion['name2'] and 
                float(values[2]) == suggestion['confidence']):
                
                self.merge_tree.selection_set(item)
                self.reject_merge()
                break
    
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

            # Clean up large data structures
            try:
                del all_embeddings
                del all_labels
                logger.debug("Cleaned up embedding structures in generate_pseudo_labels")
            except NameError:
                logger.debug("Some embedding structures were not defined in generate_pseudo_labels.")
                pass
            
        except Exception as e:
            messagebox.showerror("Error", f"Pseudo-label generation failed: {str(e)}")
            self.status_var.set("Pseudo-label generation failed")
    
    def mark_outlier_correct(self):
        """Mark selected outlier as correctly labeled."""
        selection = self.outlier_listbox.curselection()
        if not selection:
            messagebox.showwarning("Warning", "Please select an outlier")
            return
        
        idx = selection[0]
        if idx >= len(self.current_outliers):
            messagebox.showerror("Error", "Invalid outlier selection")
            return
            
        outlier = self.current_outliers[idx]
        crow_id = outlier['metadata']['crow_id']
        
        try:
            # Mark this outlier as verified in database
            # We can add a verification flag to the crow_embeddings table or create a separate verification table
            conn = get_connection()
            cursor = conn.cursor()
            
            # For now, we'll update the confidence score to indicate it's been manually verified
            # In a more complete implementation, you might add a 'verified' column
            cursor.execute("""
                UPDATE crow_embeddings 
                SET confidence = CASE 
                    WHEN confidence < 0.95 THEN 0.95 
                    ELSE confidence 
                END
                WHERE crow_id = ? AND image_path = ?
            """, (crow_id, outlier['metadata']['image_path']))
            
            conn.commit()
            conn.close()
            
            # Remove from outlier list
            self.outlier_listbox.delete(idx)
            self.current_outliers.pop(idx)
            
            messagebox.showinfo("Success", f"Outlier for {outlier['metadata']['crow_name']} marked as correctly labeled")
            
        except Exception as e:
            logger.error(f"Error marking outlier as correct: {e}")
            messagebox.showerror("Error", f"Failed to mark outlier as correct: {str(e)}")
    
    def relabel_outlier(self):
        """Relabel selected outlier."""
        selection = self.outlier_listbox.curselection()
        if not selection:
            messagebox.showwarning("Warning", "Please select an outlier")
            return
        
        idx = selection[0]
        if idx >= len(self.current_outliers):
            messagebox.showerror("Error", "Invalid outlier selection")
            return
            
        outlier = self.current_outliers[idx]
        self.open_relabel_dialog(outlier, idx)
    
    def open_relabel_dialog(self, outlier, outlier_idx):
        """Open dialog to relabel an outlier to a different crow."""
        dialog = tk.Toplevel(self.master)
        dialog.title("Relabel Outlier")
        dialog.geometry("400x300")
        dialog.transient(self.master)
        dialog.grab_set()
        
        # Header info
        header_frame = ttk.Frame(dialog)
        header_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Label(header_frame, text=f"Relabeling outlier from: {outlier['metadata']['crow_name']}", 
                 font=('Arial', 10, 'bold')).pack()
        ttk.Label(header_frame, text=f"Image: {Path(outlier['metadata']['image_path']).name}").pack()
        
        # Crow selection
        selection_frame = ttk.Frame(dialog)
        selection_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        ttk.Label(selection_frame, text="Select new crow ID:").pack(anchor='w')
        
        # Get all available crows
        try:
            crows = get_all_crows()
            crow_options = [f"{crow['id']}: {crow.get('name', 'Crow_' + str(crow['id']))}" for crow in crows]
            
            crow_var = tk.StringVar()
            crow_listbox = tk.Listbox(selection_frame, height=8)
            for option in crow_options:
                crow_listbox.insert(tk.END, option)
            crow_listbox.pack(fill='both', expand=True, pady=5)
            
            # Option to create new crow
            new_crow_frame = ttk.Frame(selection_frame)
            new_crow_frame.pack(fill='x', pady=5)
            
            create_new_var = tk.BooleanVar()
            ttk.Checkbutton(new_crow_frame, text="Create new crow", 
                           variable=create_new_var).pack(side='left')
            
            new_name_var = tk.StringVar()
            ttk.Entry(new_crow_frame, textvariable=new_name_var, 
                     placeholder_text="New crow name").pack(side='right', fill='x', expand=True, padx=(10, 0))
            
        except Exception as e:
            ttk.Label(selection_frame, text=f"Error loading crows: {str(e)}").pack()
            return
        
        # Buttons
        button_frame = ttk.Frame(dialog)
        button_frame.pack(fill='x', padx=10, pady=5)
        
        def apply_relabel():
            try:
                if create_new_var.get():
                    # Create new crow
                    new_name = new_name_var.get().strip()
                    if not new_name:
                        messagebox.showerror("Error", "Please enter a name for the new crow")
                        return
                    
                    # Create new crow in database
                    conn = get_connection()
                    cursor = conn.cursor()
                    cursor.execute("INSERT INTO crows (name) VALUES (?)", (new_name,))
                    new_crow_id = cursor.lastrowid
                    conn.commit()
                    conn.close()
                    
                    target_crow_id = new_crow_id
                    target_name = new_name
                else:
                    # Use selected existing crow
                    selection = crow_listbox.curselection()
                    if not selection:
                        messagebox.showerror("Error", "Please select a crow or create a new one")
                        return
                    
                    selected_text = crow_listbox.get(selection[0])
                    target_crow_id = int(selected_text.split(':')[0])
                    target_name = selected_text.split(':', 1)[1].strip()
                
                # Move the embedding to the new crow
                old_crow_id = outlier['metadata']['crow_id']
                image_path = outlier['metadata']['image_path']
                
                # Get the specific embedding ID
                conn = get_connection()
                cursor = conn.cursor()
                cursor.execute("SELECT id FROM crow_embeddings WHERE crow_id = ? AND image_path = ?", 
                             (old_crow_id, image_path))
                result = cursor.fetchone()
                
                if result:
                    embedding_id = result[0]
                    # Use the existing reassign function
                    moved_count = reassign_crow_embeddings(old_crow_id, target_crow_id, [embedding_id])
                    
                    if moved_count > 0:
                        # Remove from outlier list
                        self.outlier_listbox.delete(outlier_idx)
                        self.current_outliers.pop(outlier_idx)
                        
                        messagebox.showinfo("Success", f"Outlier relabeled to {target_name}")
                        dialog.destroy()
                    else:
                        messagebox.showerror("Error", "Failed to reassign embedding")
                else:
                    messagebox.showerror("Error", "Could not find embedding in database")
                
                conn.close()
                
            except Exception as e:
                logger.error(f"Error relabeling outlier: {e}")
                messagebox.showerror("Error", f"Failed to relabel outlier: {str(e)}")
        
        ttk.Button(button_frame, text="Apply", command=apply_relabel).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Cancel", command=dialog.destroy).pack(side='right', padx=5)
    
    def mark_not_crow(self):
        """Mark selected outlier as not a crow."""
        selection = self.outlier_listbox.curselection()
        if not selection:
            messagebox.showwarning("Warning", "Please select an outlier")
            return
        
        idx = selection[0]
        if idx >= len(self.current_outliers):
            messagebox.showerror("Error", "Invalid outlier selection")
            return
            
        outlier = self.current_outliers[idx]
        
        # Confirm the action
        result = messagebox.askyesno(
            "Confirm Action",
            f"Mark this image as 'not a crow'?\n"
            f"Crow: {outlier['metadata']['crow_name']}\n"
            f"Image: {Path(outlier['metadata']['image_path']).name}\n\n"
            f"This will remove the embedding from the database."
        )
        
        if not result:
            return
            
        try:
            # Remove the embedding from the database
            crow_id = outlier['metadata']['crow_id']
            image_path = outlier['metadata']['image_path']
            
            conn = get_connection()
            cursor = conn.cursor()
            
            # Delete the specific embedding
            cursor.execute("DELETE FROM crow_embeddings WHERE crow_id = ? AND image_path = ?", 
                         (crow_id, image_path))
            
            deleted_count = cursor.rowcount
            
            if deleted_count > 0:
                # Update crow sighting count
                cursor.execute("""
                    UPDATE crows 
                    SET total_sightings = (
                        SELECT COUNT(*) FROM crow_embeddings WHERE crow_id = ?
                    )
                    WHERE id = ?
                """, (crow_id, crow_id))
                
                conn.commit()
                
                # Remove from outlier list
                self.outlier_listbox.delete(idx)
                self.current_outliers.pop(idx)
                
                messagebox.showinfo("Success", f"Embedding marked as 'not a crow' and removed from database")
            else:
                messagebox.showwarning("Warning", "No embedding found to delete")
            
            conn.close()
            
        except Exception as e:
            logger.error(f"Error marking as not crow: {e}")
            messagebox.showerror("Error", f"Failed to mark as not crow: {str(e)}")


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