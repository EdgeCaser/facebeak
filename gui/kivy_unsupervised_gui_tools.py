import os
import logging
from pathlib import Path
import json
import threading
from typing import Tuple, Dict
import numpy as np # For ClusteringBasedLabelSmoother
import cv2 # For image loading if not directly handled by Kivy Image for all cases
from PIL import Image as PILImage # For image manipulation if needed

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.scrollview import ScrollView
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.slider import Slider
from kivy.uix.image import Image as KivyImage
from kivy.uix.tabbedpanel import TabbedPanel, TabbedPanelItem
from kivy.uix.popup import Popup
from kivy.uix.recycleview import RecycleView
from kivy.uix.recycleview.views import RecycleDataViewBehavior
from kivy.properties import StringProperty, ObjectProperty, ListProperty, BooleanProperty, NumericProperty
from kivy.core.window import Window
from kivy.clock import Clock
from kivy.uix.spinner import Spinner
from kivy.uix.progressbar import ProgressBar
from kivy.uix.checkbox import CheckBox
from kivy.uix.togglebutton import ToggleButton

# Assuming db.py and other necessary modules are in the same directory or accessible
from db import get_all_crows, get_crow_embeddings, update_crow_name, reassign_crow_embeddings, get_connection
from unsupervised_learning import (
    UnsupervisedTrainingPipeline, # May not be directly used by GUI, but smoother might use its concepts
    AutoLabelingSystem, 
    ReconstructionValidator
)
from crow_clustering import CrowClusterAnalyzer # Used by smoother
# from models import CrowResNetEmbedder # May be needed by smoother if it reloads models

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Preserve the ClusteringBasedLabelSmoother class (with minor adaptations if needed for Kivy)
class ClusteringBasedLabelSmoother:
    """Implements clustering-based label smoothing with GUI review."""
    
    def __init__(self, confidence_threshold: float = 0.8):
        self.confidence_threshold = confidence_threshold
        self.analyzer = CrowClusterAnalyzer() # Assuming this is Kivy-agnostic
        self.auto_labeler = AutoLabelingSystem() # Assuming this is Kivy-agnostic

    def perform_merge_operation(self, crow_id_from: int, crow_id_to: int) -> Tuple[bool, str]:
        logger.info(f"Attempting to merge crow {crow_id_from} into crow {crow_id_to}")
        try:
            moved_count = reassign_crow_embeddings(from_crow_id=crow_id_from, to_crow_id=crow_id_to)
            logger.info(f"Reassigned {moved_count} embeddings from crow {crow_id_from} to {crow_id_to}")
            conn = None
            try:
                conn = get_connection()
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM crow_embeddings WHERE crow_id = ?", (crow_id_from,))
                remaining_embeddings = cursor.fetchone()[0]
                if remaining_embeddings > 0:
                    logger.warning(f"Crow {crow_id_from} still has {remaining_embeddings} embeddings. Deleting them now.")
                    cursor.execute("DELETE FROM crow_embeddings WHERE crow_id = ?", (crow_id_from,))
                cursor.execute("DELETE FROM crows WHERE id = ?", (crow_id_from,))
                conn.commit()
                logger.info(f"Successfully deleted crow {crow_id_from} from crows table.")
            except Exception as e_db_delete:
                if conn: conn.rollback()
                logger.error(f"Database error while deleting crow {crow_id_from}: {e_db_delete}")
                return False, f"Failed to delete crow {crow_id_from}: {e_db_delete}"
            finally:
                if conn: conn.close()
            return True, f"Successfully merged crow {crow_id_from} into {crow_id_to}."
        except Exception as e:
            logger.error(f"Error during merge operation for {crow_id_from} -> {crow_id_to}: {e}")
            return False, f"Merge failed: {str(e)}"
        
    def analyze_and_suggest_merges(self) -> Dict:
        logger.info("🔍 Analyzing embeddings for potential merges...")
        crows = get_all_crows()
        crow_embeddings_data = {}
        for crow in crows:
            embeddings = get_crow_embeddings(crow['id'])
            if embeddings:
                crow_embeddings_data[crow['id']] = {
                    'name': crow.get('name', f'Crow_{crow["id"]}'),
                    'embeddings': [emb['embedding'] for emb in embeddings],
                    'image_paths': [emb.get('image_path', '') for emb in embeddings] # Keep image paths
                }
        if len(crow_embeddings_data) < 2:
            return {'suggestions': [], 'message': 'Need at least 2 crows for merge analysis', 'analysis_complete': True}
        
        merge_suggestions = []
        crow_ids = list(crow_embeddings_data.keys())
        for i in range(len(crow_ids)):
            for j in range(i + 1, len(crow_ids)):
                crow_id1, data1 = crow_ids[i], crow_embeddings_data[crow_ids[i]]
                crow_id2, data2 = crow_ids[j], crow_embeddings_data[crow_ids[j]]
                
                similarities = []
                if not data1['embeddings'] or not data2['embeddings']: continue
                for emb1 in data1['embeddings']:
                    for emb2 in data2['embeddings']:
                        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
                        similarities.append(similarity)
                
                if not similarities: continue
                avg_similarity = np.mean(similarities)
                
                if avg_similarity > self.confidence_threshold:
                    merge_suggestions.append({
                        'crow_id1': crow_id1, 'crow_id2': crow_id2,
                        'name1': data1['name'], 'name2': data2['name'],
                        'avg_similarity': float(avg_similarity),
                        'confidence': float(avg_similarity), # Redundant but kept for compatibility
                        'n_comparisons': len(similarities),
                        # Store some sample image paths for quick review popup
                        'sample_images1': data1['image_paths'][:3], 
                        'sample_images2': data2['image_paths'][:3]
                    })
        merge_suggestions.sort(key=lambda x: x['confidence'], reverse=True)
        logger.info(f"Found {len(merge_suggestions)} potential merges")
        return {'suggestions': merge_suggestions, 'total_crows': len(crow_embeddings_data), 'analysis_complete': True}
    
    def suggest_outlier_relabeling(self) -> Dict:
        logger.info("🔍 Analyzing outliers for potential relabeling...")
        crows = get_all_crows()
        all_embeddings_list = []
        all_labels_list = []
        all_metadata_list = []
        for crow in crows:
            embeddings = get_crow_embeddings(crow['id'])
            for emb in embeddings:
                all_embeddings_list.append(emb['embedding'])
                all_labels_list.append(crow['id']) # Original label
                all_metadata_list.append({
                    'crow_id': crow['id'], 
                    'crow_name': crow.get('name', f'Crow_{crow["id"]}'),
                    'image_path': emb.get('image_path', ''),
                    'embedding_id': emb.get('id') # Store embedding_id
                })
        if len(all_embeddings_list) < 10: # Arbitrary minimum
            return {'outliers': [], 'message': 'Need more embeddings for robust outlier analysis'}

        import torch # Local import for torch usage
        embeddings_tensor = torch.tensor(all_embeddings_list, dtype=torch.float32)
        validator = ReconstructionValidator() # Assuming this is Kivy-agnostic
        validator.train_autoencoder(embeddings_tensor) # This could be long
        outlier_indices, reconstruction_errors = validator.detect_outliers_with_errors(embeddings_tensor)
        
        outliers = [{'index': int(idx), 'metadata': all_metadata_list[idx], 'reconstruction_error': float(reconstruction_errors[idx])} for idx in outlier_indices]
        logger.info(f"Found {len(outliers)} potential outliers")
        return {'outliers': outliers, 'total_samples': len(all_embeddings_list)}

    def generate_pseudo_labels(self, confidence_threshold=0.95):
        logger.info(f"Generating pseudo labels with threshold {confidence_threshold}")
        # This is a simplified version. The original uses AutoLabelingSystem.
        # For Kivy port, we'll assume a similar structure.
        # The actual heavy lifting of training/predicting would be in AutoLabelingSystem
        # For the GUI, we mainly care about triggering it and displaying results.
        
        # Placeholder for demonstration
        # In a real scenario, this would involve:
        # 1. Loading unlabeled data (embeddings)
        # 2. Using a trained model (from AutoLabelingSystem) to predict labels
        # 3. Filtering by confidence
        
        # Mock results
        mock_pseudo_labels = {
            0: {'label': 'Crow_A', 'confidence': 0.98, 'image_path': 'path/to/img1.jpg'},
            1: {'label': 'Crow_B', 'confidence': 0.96, 'image_path': 'path/to/img2.jpg'}
        }
        num_processed = 100
        
        return {
            'pseudo_labels': mock_pseudo_labels,
            'num_processed': num_processed,
            'threshold_used': confidence_threshold
        }


class UnsupervisedToolsLayout(BoxLayout):
    def __init__(self, app, **kwargs):
        super().__init__(**kwargs)
        self.app = app
        self.orientation = 'vertical'
        self.padding = "10dp"
        self.spacing = "5dp"

        # Tabbed Panel
        self.tab_panel = TabbedPanel(do_default_tab=True, tab_pos='top_left')

        # Merge Suggestions Tab
        self.merge_tab = TabbedPanelItem(text="Merge Suggestions")
        self.merge_tab.content = self._create_merge_tab_layout()
        self.tab_panel.add_widget(self.merge_tab)

        # Outlier Review Tab
        self.outlier_tab = TabbedPanelItem(text="Outlier Review")
        self.outlier_tab.content = self._create_outlier_tab_layout()
        self.tab_panel.add_widget(self.outlier_tab)
        
        # Auto-Labeling Tab
        self.auto_label_tab = TabbedPanelItem(text="Auto-Labeling")
        self.auto_label_tab.content = self._create_auto_label_tab_layout()
        self.tab_panel.add_widget(self.auto_label_tab)

        self.add_widget(self.tab_panel)

        # Status Bar
        self.status_label = Label(text="Ready", size_hint_y=None, height="30dp")
        self.add_widget(self.status_label)

    def _create_merge_tab_layout(self):
        layout = BoxLayout(orientation='vertical', spacing="5dp", padding="5dp")
        # Controls
        controls = BoxLayout(size_hint_y=None, height="40dp", spacing="10dp")
        controls.add_widget(Button(text="Analyze Merges", on_press=self.app.analyze_merges_action))
        controls.add_widget(Label(text="Min Confidence:"))
        self.merge_confidence_slider = Slider(min=0.5, max=0.99, value=0.8, step=0.01)
        self.merge_confidence_label = Label(text=f"{self.merge_confidence_slider.value:.2f}")
        self.merge_confidence_slider.bind(value=lambda instance, value: setattr(self.merge_confidence_label, 'text', f"{value:.2f}"))
        controls.add_widget(self.merge_confidence_slider)
        controls.add_widget(self.merge_confidence_label)
        layout.add_widget(controls)
        
        # Results - Placeholder for RecycleView/ScrollView
        self.merge_results_layout = GridLayout(cols=1, spacing="2dp", size_hint_y=None)
        self.merge_results_layout.bind(minimum_height=self.merge_results_layout.setter('height'))
        scroll_view_merges = ScrollView()
        scroll_view_merges.add_widget(self.merge_results_layout)
        layout.add_widget(scroll_view_merges)
        
        # Actions
        actions = BoxLayout(size_hint_y=None, height="40dp", spacing="10dp")
        actions.add_widget(Button(text="Apply Selected Merge", on_press=self.app.apply_merge_action))
        actions.add_widget(Button(text="Reject Selected", on_press=self.app.reject_merge_action))
        actions.add_widget(Button(text="Review Images", on_press=self.app.review_merge_images_action))
        layout.add_widget(actions)
        return layout

    def _create_outlier_tab_layout(self):
        layout = BoxLayout(orientation='vertical', spacing="5dp", padding="5dp")
        # Controls
        controls = BoxLayout(size_hint_y=None, height="40dp")
        controls.add_widget(Button(text="Find Outliers", on_press=self.app.find_outliers_action))
        layout.add_widget(controls)

        # Results (Outlier List on Left, Image + Actions on Right)
        results_area = BoxLayout(orientation='horizontal', spacing="10dp")
        
        # Left: Outlier List
        self.outlier_list_layout = GridLayout(cols=1, spacing="2dp", size_hint_x=0.4, size_hint_y=None)
        self.outlier_list_layout.bind(minimum_height=self.outlier_list_layout.setter('height'))
        scroll_view_outliers = ScrollView(size_hint_x=0.4)
        scroll_view_outliers.add_widget(self.outlier_list_layout)
        results_area.add_widget(scroll_view_outliers)

        # Right: Image Display + Actions
        outlier_display_actions = BoxLayout(orientation='vertical', size_hint_x=0.6, spacing="5dp")
        self.outlier_image = KivyImage(source='', allow_stretch=True, keep_ratio=True)
        outlier_display_actions.add_widget(self.outlier_image)
        
        outlier_buttons = BoxLayout(size_hint_y=None, height="40dp", spacing="5dp")
        outlier_buttons.add_widget(Button(text="Mark as Correct", on_press=self.app.mark_outlier_correct_action))
        outlier_buttons.add_widget(Button(text="Relabel Crow", on_press=self.app.relabel_outlier_action))
        outlier_buttons.add_widget(Button(text="Mark as Not Crow", on_press=self.app.mark_outlier_not_crow_action))
        outlier_display_actions.add_widget(outlier_buttons)
        results_area.add_widget(outlier_display_actions)
        
        layout.add_widget(results_area)
        return layout

    def _create_auto_label_tab_layout(self):
        layout = BoxLayout(orientation='vertical', spacing="5dp", padding="5dp")
        # Controls
        controls = BoxLayout(size_hint_y=None, height="40dp", spacing="10dp")
        controls.add_widget(Button(text="Generate Pseudo-Labels", on_press=self.app.generate_pseudo_labels_action))
        controls.add_widget(Label(text="Confidence Threshold:"))
        self.auto_label_confidence_slider = Slider(min=0.8, max=0.99, value=0.95, step=0.01)
        self.auto_label_confidence_label = Label(text=f"{self.auto_label_confidence_slider.value:.2f}")
        self.auto_label_confidence_slider.bind(value=lambda instance, value: setattr(self.auto_label_confidence_label, 'text', f"{value:.2f}"))
        controls.add_widget(self.auto_label_confidence_slider)
        controls.add_widget(self.auto_label_confidence_label)
        layout.add_widget(controls)
        
        # Results Display
        self.auto_label_results_text = TextInput(readonly=True, multiline=True, hint_text="Auto-labeling results will appear here.")
        scroll_view_auto_label = ScrollView()
        scroll_view_auto_label.add_widget(self.auto_label_results_text)
        layout.add_widget(scroll_view_auto_label)
        return layout


class UnsupervisedToolsApp(App):
    def build(self):
        self.title = "Facebeak - Unsupervised Learning Tools (Kivy)"
        Window.size = (1200, 800)
        
        self.label_smoother = ClusteringBasedLabelSmoother() # Initialize backend logic
        self.current_merge_suggestions = [] # To store full suggestion data
        self.selected_merge_suggestion_idx = None # To track selection in merge list
        self.current_outliers_data = [] # To store full outlier data
        self.selected_outlier_idx = None # To track selection in outlier list
        
        self.layout = UnsupervisedToolsLayout(app=self)
        return self.layout

    def on_start(self):
        logger.info("UnsupervisedToolsApp started.")
        self.layout.status_label.text = "Ready. Select an action from one of the tabs."

    # --- Merge Suggestions Tab Actions ---
    def analyze_merges_action(self, instance):
        self.layout.status_label.text = "Analyzing potential merges... Please wait."
        confidence = self.layout.merge_confidence_slider.value
        self.label_smoother.confidence_threshold = confidence
        
        # Run in thread to avoid freezing GUI
        threading.Thread(target=self._do_analyze_merges, daemon=True).start()

    def _do_analyze_merges(self):
        try:
            results = self.label_smoother.analyze_and_suggest_merges()
            Clock.schedule_once(lambda dt: self._update_merge_suggestions_ui(results))
        except Exception as e:
            logger.error(f"Error during merge analysis: {e}", exc_info=True)
            Clock.schedule_once(lambda dt: self.show_popup("Error", f"Merge analysis failed: {e}"))
            Clock.schedule_once(lambda dt: setattr(self.layout.status_label, 'text', "Merge analysis failed."))

    def _update_merge_suggestions_ui(self, results):
        self.layout.merge_results_layout.clear_widgets()
        self.current_merge_suggestions = results.get('suggestions', [])
        
        if not self.current_merge_suggestions:
            msg = results.get('message', "No merge suggestions found or analysis failed.")
            self.layout.merge_results_layout.add_widget(Label(text=msg))
            self.layout.status_label.text = msg
            return

        for idx, suggestion in enumerate(self.current_merge_suggestions):
            # Create a more interactive item for selection
            item_text = (f"Merge Crow {suggestion['crow_id1']} ({suggestion['name1']}) with "
                         f"Crow {suggestion['crow_id2']} ({suggestion['name2']})\n"
                         f"Confidence: {suggestion['confidence']:.3f} (Based on {suggestion['n_comparisons']} comparisons)")
            
            # Using ListItemButton for simplicity, can be customized
            btn = ListItemButton(text=item_text, size_hint_y=None, height="60dp")
            btn.suggestion_idx = idx # Store index for later retrieval
            btn.bind(on_press=self.on_merge_suggestion_select)
            self.layout.merge_results_layout.add_widget(btn)
        
        self.layout.status_label.text = f"Found {len(self.current_merge_suggestions)} merge suggestions."
        self.selected_merge_suggestion_idx = None 
        # Deselect all buttons visually
        for child in self.layout.merge_results_layout.children:
            if isinstance(child, ListItemButton): # Or your custom item widget
                child.deselect()


    def on_merge_suggestion_select(self, instance_button):
        # Deselect other buttons if they are part of a selection group or manage manually
        for child in self.layout.merge_results_layout.children:
            if isinstance(child, ListItemButton) and child != instance_button:
                child.is_selected = False # Kivy ListItemButton has is_selected
        instance_button.is_selected = True

        self.selected_merge_suggestion_idx = instance_button.suggestion_idx
        logger.info(f"Selected merge suggestion index: {self.selected_merge_suggestion_idx}")


    def apply_merge_action(self, instance):
        if self.selected_merge_suggestion_idx is None or \
           not (0 <= self.selected_merge_suggestion_idx < len(self.current_merge_suggestions)):
            self.show_popup("Selection Error", "Please select a valid merge suggestion first.")
            return
        suggestion = self.current_merge_suggestions[self.selected_merge_suggestion_idx]
        
        # Confirmation Popup
        content = BoxLayout(orientation='vertical', padding="10dp", spacing="10dp")
        # Ensure the label text wraps
        confirm_label = Label(text=f"Merge '{suggestion['name1']}' (ID {suggestion['crow_id1']}) into '{suggestion['name2']}' (ID {suggestion['crow_id2']})?\nConfidence: {suggestion['confidence']:.3f}\nThis action cannot be easily undone.", 
                              text_size=(Window.width*0.45, None), halign='center')
        content.add_widget(confirm_label)
        
        buttons = BoxLayout(size_hint_y=None, height="44dp", spacing="10dp")
        confirm_btn = Button(text="Confirm Merge")
        cancel_btn = Button(text="Cancel")
        buttons.add_widget(confirm_btn)
        buttons.add_widget(cancel_btn)
        content.add_widget(buttons)
        
        popup_height = confirm_label.texture_size[1] + buttons.height + 60 # Calculate required height based on text
        popup = Popup(title="Confirm Merge", content=content, size_hint=(0.5, None), height=f"{popup_height}dp", auto_dismiss=False)
        
        def do_merge(btn_instance):
            popup.dismiss()
            self.layout.status_label.text = f"Merging {suggestion['name1']} into {suggestion['name2']}..."
            threading.Thread(target=self._do_apply_merge, args=(suggestion,), daemon=True).start()

        confirm_btn.bind(on_press=do_merge)
        cancel_btn.bind(on_press=popup.dismiss)
        popup.open()

    def _do_apply_merge(self, suggestion):
        try:
            crow_id_from, crow_id_to = suggestion['crow_id1'], suggestion['crow_id2']
            # Optional: Add logic to always merge smaller ID into larger, or based on other criteria
            # if crow_id_from > crow_id_to:
            #     crow_id_from, crow_id_to = crow_id_to, crow_id_from
            #     logger.info(f"Swapped merge order: {crow_id_from} (was {suggestion['crow_id2']}) into {crow_id_to} (was {suggestion['crow_id1']})")

            success, message = self.label_smoother.perform_merge_operation(crow_id_from, crow_id_to)
            
            if success:
                Clock.schedule_once(lambda dt: self.show_popup("Success", message))
                Clock.schedule_once(lambda dt: setattr(self.layout.status_label, 'text', "Merge completed. Re-analyzing suggestions."))
                Clock.schedule_once(lambda dt: self.analyze_merges_action(None)) 
            else:
                Clock.schedule_once(lambda dt: self.show_popup("Error", f"Merge failed: {message}"))
                Clock.schedule_once(lambda dt: setattr(self.layout.status_label, 'text', "Merge failed."))
        except Exception as e:
            logger.error(f"Error applying merge: {e}", exc_info=True)
            Clock.schedule_once(lambda dt: self.show_popup("Error", f"Merge application process failed: {e}"))

    def reject_merge_action(self, instance):
        if self.selected_merge_suggestion_idx is None or \
           not (0 <= self.selected_merge_suggestion_idx < len(self.current_merge_suggestions)):
            self.show_popup("Info", "No merge suggestion selected to reject.")
            return
        
        suggestion = self.current_merge_suggestions[self.selected_merge_suggestion_idx]
        logger.info(f"Rejected merge suggestion: Crow {suggestion['crow_id1']} and Crow {suggestion['crow_id2']}")
        self.show_popup("Info", f"Merge for Crow {suggestion['crow_id1']} & {suggestion['crow_id2']} marked as rejected (UI only for now).")
        
        # Visually update the item in the list (e.g., disable it or change text)
        # This requires accessing the specific widget in merge_results_layout
        # For ListItemButton, we might change its text or disabled state.
        # This is a bit tricky if not using RecycleView with data binding.
        # For now, we'll just clear the selection.
        if self.layout.merge_results_layout.children: # Children are added in reverse order by Kivy
            target_button_index_from_top = self.selected_merge_suggestion_idx
            # Kivy adds widgets to children list such that children[0] is the last added.
            # If items are added sequentially, the item at suggestion_idx is at:
            # len(self.layout.merge_results_layout.children) - 1 - target_button_index_from_top
            actual_child_idx = len(self.layout.merge_results_layout.children) - 1 - target_button_index_from_top
            if 0 <= actual_child_idx < len(self.layout.merge_results_layout.children):
                button_to_update = self.layout.merge_results_layout.children[actual_child_idx]
                button_to_update.text = "[REJECTED] " + button_to_update.text
                button_to_update.disabled = True # Disable further interaction
        
        self.selected_merge_suggestion_idx = None 

    def review_merge_images_action(self, instance):
        if self.selected_merge_suggestion_idx is None or \
           not (0 <= self.selected_merge_suggestion_idx < len(self.current_merge_suggestions)):
            self.show_popup("Selection Error", "Please select a merge suggestion to review.")
            return
        suggestion = self.current_merge_suggestions[self.selected_merge_suggestion_idx]
        self._show_merge_review_popup(suggestion)

    def _show_merge_review_popup(self, suggestion):
        content = BoxLayout(orientation='vertical', padding="10dp", spacing="5dp")
        title_text = (f"Review: Merge Crow {suggestion['crow_id1']} ({suggestion['name1']}) "
                      f"with Crow {suggestion['crow_id2']} ({suggestion['name2']})? "
                      f"Conf: {suggestion['confidence']:.3f}")
        content.add_widget(Label(text=title_text, size_hint_y=None, height="40dp", font_size='16sp'))

        images_layout = GridLayout(cols=2, spacing="10dp")
        
        # Crow 1 Images
        crow1_panel = BoxLayout(orientation='vertical')
        crow1_panel.add_widget(Label(text=f"Crow {suggestion['crow_id1']}: {suggestion['name1']}", size_hint_y=None, height="20dp"))
        scroll1 = ScrollView()
        grid1 = GridLayout(cols=1, spacing="2dp", size_hint_y=None) # Display images vertically for now
        grid1.bind(minimum_height=grid1.setter('height'))
        for img_path in suggestion.get('sample_images1', []):
            if os.path.exists(img_path):
                grid1.add_widget(KivyImage(source=img_path, size_hint_y=None, height="100dp", allow_stretch=True, keep_ratio=True))
            else:
                grid1.add_widget(Label(text=f"Img not found:\n{os.path.basename(img_path)}", size_hint_y=None, height="100dp"))
        scroll1.add_widget(grid1)
        crow1_panel.add_widget(scroll1)
        images_layout.add_widget(crow1_panel)

        # Crow 2 Images
        crow2_panel = BoxLayout(orientation='vertical')
        crow2_panel.add_widget(Label(text=f"Crow {suggestion['crow_id2']}: {suggestion['name2']}", size_hint_y=None, height="20dp"))
        scroll2 = ScrollView()
        grid2 = GridLayout(cols=1, spacing="2dp", size_hint_y=None)
        grid2.bind(minimum_height=grid2.setter('height'))
        for img_path in suggestion.get('sample_images2', []):
            if os.path.exists(img_path):
                grid2.add_widget(KivyImage(source=img_path, size_hint_y=None, height="100dp", allow_stretch=True, keep_ratio=True))
            else:
                grid2.add_widget(Label(text=f"Img not found:\n{os.path.basename(img_path)}", size_hint_y=None, height="100dp"))
        scroll2.add_widget(grid2)
        crow2_panel.add_widget(scroll2)
        images_layout.add_widget(crow2_panel)
        
        content.add_widget(images_layout)

        buttons = BoxLayout(size_hint_y=None, height="44dp", spacing="10dp")
        merge_btn = Button(text="Confirm Merge This Pair")
        keep_btn = Button(text="Keep Separate (Reject This)")
        close_btn = Button(text="Just Close Review")
        buttons.add_widget(merge_btn)
        buttons.add_widget(keep_btn)
        buttons.add_widget(close_btn)
        content.add_widget(buttons)

        popup = Popup(title="Review Merge Suggestion", content=content, size_hint=(0.9, 0.9), auto_dismiss=True)
        
        def _confirm_merge_from_review(btn_instance):
            popup.dismiss()
            self.apply_merge_action(None) # apply_merge_action uses self.selected_merge_suggestion_idx

        def _reject_merge_from_review(btn_instance):
            popup.dismiss()
            self.reject_merge_action(None) # reject_merge_action uses self.selected_merge_suggestion_idx

        merge_btn.bind(on_press=_confirm_merge_from_review)
        keep_btn.bind(on_press=_reject_merge_from_review)
        close_btn.bind(on_press=popup.dismiss)
        popup.open()


    # --- Outlier Review Tab Actions ---
    def find_outliers_action(self, instance):
        self.layout.status_label.text = "Finding outliers... Please wait."
        # Clear previous selection and image
        self.selected_outlier_idx = None
        self.layout.outlier_image.source = ''
        threading.Thread(target=self._do_find_outliers, daemon=True).start()

    def _do_find_outliers(self):
        try:
            results = self.label_smoother.suggest_outlier_relabeling()
            Clock.schedule_once(lambda dt: self._update_outliers_ui(results))
        except Exception as e:
            logger.error(f"Error during outlier detection: {e}", exc_info=True)
            Clock.schedule_once(lambda dt: self.show_popup("Error", f"Outlier detection failed: {e}"))
            Clock.schedule_once(lambda dt: setattr(self.layout.status_label, 'text', "Outlier detection failed."))

    def _update_outliers_ui(self, results):
        self.layout.outlier_list_layout.clear_widgets()
        self.layout.outlier_image.source = '' 
        self.current_outliers_data = results.get('outliers', [])

        if not self.current_outliers_data:
            msg = results.get('message', "No outliers found or analysis failed.")
            self.layout.outlier_list_layout.add_widget(Label(text=msg, size_hint_y=None, height='30dp'))
            self.layout.status_label.text = msg
            return

        for idx, outlier_data in enumerate(self.current_outliers_data):
            metadata = outlier_data['metadata']
            item_text = (f"Crow {metadata['crow_id']} ({metadata['crow_name']})\n"
                         f"{os.path.basename(metadata['image_path'])} (Err: {outlier_data['reconstruction_error']:.3f})")
            btn = ListItemButton(text=item_text, size_hint_y=None, height="60dp")
            btn.outlier_idx = idx
            btn.bind(on_press=self.on_outlier_select)
            self.layout.outlier_list_layout.add_widget(btn)
        
        self.layout.status_label.text = f"Found {len(self.current_outliers_data)} potential outliers."
        self.selected_outlier_idx = None

    def on_outlier_select(self, instance_button):
        self.selected_outlier_idx = instance_button.outlier_idx
        outlier = self.current_outliers_data[self.selected_outlier_idx]
        image_path = outlier['metadata']['image_path']
        if os.path.exists(image_path):
            self.layout.outlier_image.source = image_path
            self.layout.outlier_image.reload()
        else:
            self.layout.outlier_image.source = ''
            self.show_popup("Image Error", f"Outlier image not found at: {image_path}")
        logger.info(f"Selected outlier: {outlier}")

    def mark_outlier_correct_action(self, instance):
        if self.selected_outlier_idx is None:
            self.show_popup("Info", "No outlier selected.")
            return
        # TODO: Implement logic (e.g., mark in DB or internal list as verified)
        outlier = self.current_outliers_data[self.selected_outlier_idx]
        logger.info(f"Marked outlier as correct: {outlier}")
        self.show_popup("Info", "Outlier marked as correct (UI only for now).")
        # Optionally remove from list or disable buttons
        # self.layout.outlier_list_layout.children[-(self.selected_outlier_idx+1)].disabled = True # Example of disabling

    def relabel_outlier_action(self, instance):
        if self.selected_outlier_idx is None:
            self.show_popup("Info", "No outlier selected.")
            return
        # TODO: Implement relabeling dialog (similar to SuspectLineup)
        outlier = self.current_outliers_data[self.selected_outlier_idx]
        logger.info(f"Relabeling outlier: {outlier}")
        self.show_popup("Info", "Relabeling logic to be implemented (needs dialog).")

    def mark_outlier_not_crow_action(self, instance):
        if self.selected_outlier_idx is None:
            self.show_popup("Info", "No outlier selected.")
            return
        # TODO: Implement logic (e.g., remove embedding via DB call)
        outlier = self.current_outliers_data[self.selected_outlier_idx]
        logger.info(f"Marking outlier as 'Not Crow': {outlier}")
        self.show_popup("Info", "Marking as 'Not Crow' logic to be implemented.")

    # --- Auto-Labeling Tab Actions ---
    def generate_pseudo_labels_action(self, instance):
        self.layout.status_label.text = "Generating pseudo-labels... Please wait."
        confidence = self.layout.auto_label_confidence_slider.value # Get value from slider
        # Clear previous results
        self.layout.auto_label_results_text.text = "Processing..."
        threading.Thread(target=self._do_generate_pseudo_labels, args=(confidence,), daemon=True).start()
        # self.layout.status_label.text = "Generating pseudo-labels... Please wait."
        # confidence = self.layout.auto_label_confidence_slider.value
        # threading.Thread(target=self._do_generate_pseudo_labels, args=(confidence,), daemon=True).start()

    def _do_generate_pseudo_labels(self, confidence_threshold):
        try:
            # This currently uses a mock implementation in ClusteringBasedLabelSmoother
            results = self.label_smoother.generate_pseudo_labels(confidence_threshold)
            Clock.schedule_once(lambda dt: self._update_pseudo_labels_ui(results))
        except Exception as e:
            logger.error(f"Error during pseudo-label generation: {e}", exc_info=True)
            Clock.schedule_once(lambda dt: self.show_popup("Error", f"Pseudo-label generation failed: {e}"))
            Clock.schedule_once(lambda dt: setattr(self.layout.status_label, 'text', "Pseudo-label generation failed."))

    def _update_pseudo_labels_ui(self, results):
        threshold_used = results.get('threshold_used', self.layout.auto_label_confidence_slider.value)
        num_processed = results.get('num_processed', 'N/A')
        pseudo_labels_dict = results.get('pseudo_labels', {})

        text_output = f"Pseudo-Labeling Results (Threshold: {threshold_used:.2f})\n"
        text_output += f"Total items processed (mocked): {num_processed}\n\n"
        
        if pseudo_labels_dict:
            for idx, info in pseudo_labels_dict.items(): # Iterate over dict items
                image_path_text = info.get('image_path', 'Unknown image path')
                text_output += (f"Image: {os.path.basename(image_path_text)}\n" # Use .get for safety
                                f"  -> Suggested Label: {info.get('label', 'N/A')} (Conf: {info.get('confidence', 0.0):.3f})\n\n")
        else:
            text_output += "No confident pseudo-labels generated at this threshold or an error occurred."
            
        self.layout.auto_label_results_text.text = text_output
        self.layout.status_label.text = f"Pseudo-label generation complete. {len(pseudo_labels_dict)} suggestions."

    # --- Utility Methods ---
    def show_popup(self, title, message):
        if not isinstance(threading.current_thread(), threading._MainThread):
             Clock.schedule_once(lambda dt: self._do_show_popup(title, message))
        else:
            self._do_show_popup(title, message)

    def _do_show_popup(self, title, message):
        content = BoxLayout(orientation='vertical', padding="10dp", spacing="10dp")
        msg_label = Label(text=message, text_size=(Window.width*0.45, None), halign='center', valign='middle')
        content.add_widget(msg_label)
        ok_button = Button(text="OK", size_hint_y=None, height="44dp")
        content.add_widget(ok_button)
        popup = Popup(title=title, content=content, size_hint=(0.5, None), height="200dp", auto_dismiss=True)
        ok_button.bind(on_press=popup.dismiss)
        popup.open()
        logger.info(f"Popup shown: '{title}'")


if __name__ == '__main__':
    # Kivy specific setup, if any (e.g. for specific renderers or config)
    # os.environ['KIVY_IMAGE'] = 'pil,sdl2' # Example
    UnsupervisedToolsApp().run()
print("UnsupervisedToolsApp finished.")
