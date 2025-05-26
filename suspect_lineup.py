import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import os
from PIL import Image, ImageTk
import logging
from pathlib import Path
from db import (get_all_crows, get_first_crow_image, get_crow_videos, 
                get_crow_images_from_video, update_crow_name, 
                reassign_crow_embeddings, create_new_crow_from_embeddings,
                get_crow_embeddings, get_embedding_ids_by_image_paths,
                delete_crow_embeddings)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SuspectLineup:
    def __init__(self, root):
        self.root = root
        self.root.title("Facebeak - Suspect Lineup")
        self.root.geometry("1400x900")
        
        # Initialize state
        self.current_crow_id = None
        self.current_crow_name = None
        self.current_videos = []
        self.selected_videos = []
        self.current_images = []
        self.image_selections = {}  # Track user selections for each image
        self.unsaved_changes = False
        
        # Create main layout
        self.create_layout()
        
        # Load initial crow list
        self.load_crow_list()
        
    def create_layout(self):
        """Create the main GUI layout based on the wireframe."""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)
        
        # Left panel for controls
        control_panel = ttk.Frame(main_frame, padding="5")
        control_panel.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        control_panel.columnconfigure(0, weight=1)
        
        # Title
        title_label = ttk.Label(control_panel, text="Select Crow ID", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, pady=(0, 10))
        
        # Crow selection
        crow_frame = ttk.Frame(control_panel)
        crow_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        crow_frame.columnconfigure(0, weight=1)
        
        self.crow_var = tk.StringVar()
        self.crow_combobox = ttk.Combobox(crow_frame, textvariable=self.crow_var, 
                                         state="readonly", width=20)
        self.crow_combobox.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 5))
        
        load_button = ttk.Button(crow_frame, text="Load", command=self.load_crow)
        load_button.grid(row=0, column=1)
        
        # Crow image display
        self.crow_image_frame = ttk.LabelFrame(control_panel, text="", padding="5")
        self.crow_image_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.crow_image_label = ttk.Label(self.crow_image_frame, text="No crow loaded")
        self.crow_image_label.grid(row=0, column=0)
        
        # Crow name entry
        name_frame = ttk.Frame(control_panel)
        name_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        name_frame.columnconfigure(0, weight=1)
        
        self.name_var = tk.StringVar()
        self.name_entry = ttk.Entry(name_frame, textvariable=self.name_var)
        self.name_entry.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        # Video file selection
        video_frame = ttk.LabelFrame(control_panel, text="Video File", padding="5")
        video_frame.grid(row=4, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        video_frame.columnconfigure(0, weight=1)
        video_frame.rowconfigure(0, weight=1)
        
        # Video listbox with scrollbar
        video_list_frame = ttk.Frame(video_frame)
        video_list_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        video_list_frame.columnconfigure(0, weight=1)
        video_list_frame.rowconfigure(0, weight=1)
        
        self.video_listbox = tk.Listbox(video_list_frame, selectmode=tk.MULTIPLE, height=8)
        self.video_listbox.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.video_listbox.bind('<<ListboxSelect>>', self.on_video_select)
        
        video_scrollbar = ttk.Scrollbar(video_list_frame, orient="vertical", 
                                       command=self.video_listbox.yview)
        video_scrollbar.grid(row=0, column=1, sticky="ns")
        self.video_listbox.configure(yscrollcommand=video_scrollbar.set)
        
        # Select button for videos
        self.select_videos_button = ttk.Button(video_frame, text="Select", command=self.select_videos, state='disabled')
        self.select_videos_button.grid(row=1, column=0, pady=(5, 0))
        
        # Status label for video selection
        self.status_label = ttk.Label(video_frame, text="Select videos to load images", foreground="gray")
        self.status_label.grid(row=2, column=0, pady=(5, 0))
        
        # Save/Discard buttons
        button_frame = ttk.Frame(control_panel)
        button_frame.grid(row=5, column=0, pady=(10, 0))
        
        self.save_button = ttk.Button(button_frame, text="Save", command=self.save_changes)
        self.save_button.grid(row=0, column=0, padx=(0, 5))
        
        self.discard_button = ttk.Button(button_frame, text="Discard", command=self.discard_changes)
        self.discard_button.grid(row=0, column=1)
        
        # Right panel for sightings
        sightings_panel = ttk.LabelFrame(main_frame, text="Sightings - confirm crow identity", 
                                        padding="5")
        sightings_panel.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(10, 0))
        sightings_panel.columnconfigure(0, weight=1)
        sightings_panel.rowconfigure(0, weight=1)
        
        # Scrollable frame for images
        self.canvas = tk.Canvas(sightings_panel, bg='white')
        self.canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.scrollbar = ttk.Scrollbar(sightings_panel, orient="vertical", 
                                      command=self.canvas.yview)
        self.scrollbar.grid(row=0, column=1, sticky="ns")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        
        # Frame inside canvas for images
        self.images_frame = ttk.Frame(self.canvas)
        self.canvas_window = self.canvas.create_window((0, 0), window=self.images_frame, 
                                                      anchor="nw")
        
        # Bind canvas resize
        self.canvas.bind('<Configure>', self.on_canvas_configure)
        self.images_frame.bind('<Configure>', self.on_frame_configure)
        
        # Bind mousewheel to canvas
        self.canvas.bind("<MouseWheel>", self.on_mousewheel)
        
        # Initially disable save/discard buttons
        self.save_button.configure(state='disabled')
        self.discard_button.configure(state='disabled')
        
    def on_canvas_configure(self, event):
        """Handle canvas resize."""
        self.canvas.itemconfig(self.canvas_window, width=event.width)
        
    def on_frame_configure(self, event):
        """Handle frame resize."""
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        
    def on_mousewheel(self, event):
        """Handle mouse wheel scrolling."""
        self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
    def load_crow_list(self):
        """Load all crows into the combobox."""
        try:
            crows = get_all_crows()
            crow_options = []
            for crow in crows:
                name = crow['name'] if crow['name'] else f"Crow{crow['id']}"
                crow_options.append(f"Crow{crow['id']} ({name})")
            
            self.crow_combobox['values'] = crow_options
            if crow_options:
                self.crow_combobox.current(0)
                
        except Exception as e:
            logger.error(f"Error loading crow list: {e}")
            messagebox.showerror("Error", f"Failed to load crow list: {str(e)}")
            
    def load_crow(self):
        """Load the selected crow and display its information."""
        try:
            selected = self.crow_var.get()
            if not selected:
                messagebox.showwarning("Warning", "Please select a crow first")
                return
                
            # Extract crow ID from selection
            crow_id = int(selected.split('Crow')[1].split(' ')[0])
            self.current_crow_id = crow_id
            
            # Get crow name
            crows = get_all_crows()
            crow_data = next((c for c in crows if c['id'] == crow_id), None)
            if crow_data:
                self.current_crow_name = crow_data['name']
                self.name_var.set(self.current_crow_name or "")
            
            # Load crow image
            self.load_crow_image()
            
            # Load videos where this crow appears
            self.load_crow_videos()
            
            # Update frame title
            name_display = self.current_crow_name or f"Crow {crow_id}"
            self.crow_image_frame.configure(text=name_display)
            
            # Clear previous selections
            self.clear_sightings()
            
        except Exception as e:
            logger.error(f"Error loading crow: {e}")
            messagebox.showerror("Error", f"Failed to load crow: {str(e)}")
            
    def load_crow_image(self):
        """Load and display the first image of the current crow."""
        try:
            if not self.current_crow_id:
                return
                
            image_path = get_first_crow_image(self.current_crow_id)
            if image_path and os.path.exists(image_path):
                # Load and resize image
                image = Image.open(image_path)
                image.thumbnail((200, 200), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(image)
                
                self.crow_image_label.configure(image=photo, text="")
                self.crow_image_label.image = photo  # Keep a reference
            else:
                self.crow_image_label.configure(image="", text="No image available")
                
        except Exception as e:
            logger.error(f"Error loading crow image: {e}")
            self.crow_image_label.configure(image="", text="Error loading image")
            
    def load_crow_videos(self):
        """Load videos where the current crow appears."""
        try:
            if not self.current_crow_id:
                return
                
            self.current_videos = get_crow_videos(self.current_crow_id)
            
            # Clear and populate video listbox
            self.video_listbox.delete(0, tk.END)
            for video in self.current_videos:
                video_name = os.path.basename(video['video_path'])
                display_text = f"{video_name} ({video['sighting_count']} sightings)"
                self.video_listbox.insert(tk.END, display_text)
                
        except Exception as e:
            logger.error(f"Error loading crow videos: {e}")
            messagebox.showerror("Error", f"Failed to load videos: {str(e)}")
            
    def on_video_select(self, event):
        """Handle video selection change."""
        try:
            selected_indices = self.video_listbox.curselection()
            
            if not selected_indices:
                # No selection - disable select button and clear info
                self.select_videos_button.configure(state='disabled')
                return
            
            # Enable select button when videos are selected
            self.select_videos_button.configure(state='normal')
            
            # Update status with selection info
            num_selected = len(selected_indices)
            if num_selected == 1:
                selected_video = self.current_videos[selected_indices[0]]
                video_name = os.path.basename(selected_video['video_path'])
                status_text = f"Selected: {video_name} ({selected_video['sighting_count']} sightings)"
            else:
                total_sightings = sum(self.current_videos[i]['sighting_count'] for i in selected_indices)
                status_text = f"Selected {num_selected} videos ({total_sightings} total sightings)"
            
            # Update status label if it exists
            if hasattr(self, 'status_label'):
                self.status_label.configure(text=status_text)
            
        except Exception as e:
            logger.error(f"Error handling video selection: {e}")
            # Don't show error dialog for selection events as they happen frequently
        
    def select_videos(self):
        """Load images from selected videos."""
        try:
            selected_indices = self.video_listbox.curselection()
            if not selected_indices:
                messagebox.showwarning("Warning", "Please select one or more videos")
                return
                
            self.selected_videos = [self.current_videos[i] for i in selected_indices]
            self.load_sightings()
            
        except Exception as e:
            logger.error(f"Error selecting videos: {e}")
            messagebox.showerror("Error", f"Failed to load video images: {str(e)}")
            
    def load_sightings(self):
        """Load and display images from selected videos."""
        try:
            # Clear previous images
            self.clear_sightings()
            
            if not self.current_crow_id or not self.selected_videos:
                return
                
            # Load images from each selected video
            all_images = []
            for video in self.selected_videos:
                video_path = video['video_path']
                images = get_crow_images_from_video(self.current_crow_id, video_path)
                for img_path in images:
                    all_images.append({
                        'path': img_path,
                        'video': os.path.basename(video_path),
                        'video_path': video_path
                    })
            
            self.current_images = all_images
            self.display_images()
            
        except Exception as e:
            logger.error(f"Error loading sightings: {e}")
            messagebox.showerror("Error", f"Failed to load sightings: {str(e)}")
            
    def display_images(self):
        """Display images in the sightings panel."""
        try:
            # Clear previous widgets
            for widget in self.images_frame.winfo_children():
                widget.destroy()
                
            self.image_selections = {}
            
            if not self.current_images:
                no_images_label = ttk.Label(self.images_frame, 
                                          text="No images found for selected videos")
                no_images_label.grid(row=0, column=0, pady=20)
                return
                
            # Display each image with radio buttons
            for i, img_data in enumerate(self.current_images):
                img_path = img_data['path']
                video_name = img_data['video']
                
                # Create frame for this image
                img_frame = ttk.Frame(self.images_frame, padding="10")
                img_frame.grid(row=i, column=0, sticky=(tk.W, tk.E), pady=5)
                img_frame.columnconfigure(0, weight=1)
                
                # Image filename label
                filename = os.path.basename(img_path)
                filename_label = ttk.Label(img_frame, text=filename, font=("Arial", 10, "bold"))
                filename_label.grid(row=0, column=0, sticky=tk.W)
                
                # Image display
                if os.path.exists(img_path):
                    try:
                        image = Image.open(img_path)
                        image.thumbnail((300, 300), Image.Resampling.LANCZOS)
                        photo = ImageTk.PhotoImage(image)
                        
                        img_label = ttk.Label(img_frame, image=photo)
                        img_label.grid(row=1, column=0, pady=(5, 10))
                        img_label.image = photo  # Keep reference
                    except Exception as e:
                        logger.error(f"Error loading image {img_path}: {e}")
                        img_label = ttk.Label(img_frame, text="Error loading image")
                        img_label.grid(row=1, column=0, pady=(5, 10))
                else:
                    img_label = ttk.Label(img_frame, text="Image not found")
                    img_label.grid(row=1, column=0, pady=(5, 10))
                
                # Radio buttons for classification
                radio_frame = ttk.Frame(img_frame)
                radio_frame.grid(row=2, column=0, sticky=tk.W)
                
                # Create variable for this image
                var = tk.StringVar(value="same_crow")  # Default to same crow
                self.image_selections[img_path] = var
                
                # Radio button options
                ttk.Radiobutton(radio_frame, text=f"This is Crow{self.current_crow_id}", 
                               variable=var, value="same_crow").grid(row=0, column=0, sticky=tk.W)
                ttk.Radiobutton(radio_frame, text="This is another crow", 
                               variable=var, value="different_crow").grid(row=1, column=0, sticky=tk.W)
                ttk.Radiobutton(radio_frame, text="This not a crow", 
                               variable=var, value="not_crow").grid(row=2, column=0, sticky=tk.W)
                ttk.Radiobutton(radio_frame, text="Multiple crows in image", 
                               variable=var, value="multi_crow").grid(row=3, column=0, sticky=tk.W)
                
                # Bind change event to track unsaved changes
                var.trace('w', self.on_selection_change)
                
                # Separator
                if i < len(self.current_images) - 1:
                    separator = ttk.Separator(self.images_frame, orient='horizontal')
                    separator.grid(row=i*2+1, column=0, sticky=(tk.W, tk.E), pady=10)
                    
            # Update scroll region
            self.images_frame.update_idletasks()
            self.canvas.configure(scrollregion=self.canvas.bbox("all"))
            
        except Exception as e:
            logger.error(f"Error displaying images: {e}")
            messagebox.showerror("Error", f"Failed to display images: {str(e)}")
            
    def on_selection_change(self, *args):
        """Handle selection change to track unsaved changes."""
        self.unsaved_changes = True
        self.save_button.configure(state='normal')
        self.discard_button.configure(state='normal')
        
    def clear_sightings(self):
        """Clear the sightings display."""
        for widget in self.images_frame.winfo_children():
            widget.destroy()
        self.current_images = []
        self.image_selections = {}
        self.unsaved_changes = False
        self.save_button.configure(state='disabled')
        self.discard_button.configure(state='disabled')
        
    def save_changes(self):
        """Save the user's selections and update the database."""
        try:
            if not self.current_crow_id or not self.image_selections:
                messagebox.showwarning("Warning", "No changes to save")
                return
                
            # Update crow name if changed
            new_name = self.name_var.get().strip()
            if new_name != (self.current_crow_name or ""):
                update_crow_name(self.current_crow_id, new_name if new_name else None)
                self.current_crow_name = new_name
                
            # Process image classifications
            images_to_reassign = []
            images_to_remove = []
            images_multi_crow = []
            
            for img_path, var in self.image_selections.items():
                selection = var.get()
                if selection == "different_crow":
                    images_to_reassign.append(img_path)
                elif selection == "not_crow":
                    images_to_remove.append(img_path)
                elif selection == "multi_crow":
                    images_multi_crow.append(img_path)
                    
            # Handle reassignments and removals
            total_to_process = images_to_reassign + images_to_remove + images_multi_crow
            if total_to_process:
                result = self.process_reassignments(images_to_reassign, images_to_remove, images_multi_crow)
                if result:
                    messagebox.showinfo("Success", 
                        f"Changes saved successfully!\n"
                        f"Images reassigned: {len(images_to_reassign)}\n"
                        f"Images marked as not-crow: {len(images_to_remove)}\n"
                        f"Images marked as multi-crow: {len(images_multi_crow)}")
                else:
                    messagebox.showerror("Error", "Failed to save some changes")
            else:
                messagebox.showinfo("Success", "Crow name updated successfully!")
                
            # Reset state
            self.unsaved_changes = False
            self.save_button.configure(state='disabled')
            self.discard_button.configure(state='disabled')
            
            # Reload the crow to reflect changes
            self.load_crow()
            
        except Exception as e:
            logger.error(f"Error saving changes: {e}")
            messagebox.showerror("Error", f"Failed to save changes: {str(e)}")
            
    def process_reassignments(self, images_to_reassign, images_to_remove, images_multi_crow):
        """Process image reassignments and removals."""
        try:
            success_count = 0
            total_operations = len(images_to_reassign) + len(images_to_remove) + len(images_multi_crow)
            
            # Handle images to remove (mark as not-crow)
            if images_to_remove:
                logger.info(f"Processing {len(images_to_remove)} images to remove")
                
                # Get embedding IDs for images to remove
                remove_embedding_map = get_embedding_ids_by_image_paths(images_to_remove)
                remove_embedding_ids = list(remove_embedding_map.values())
                
                if remove_embedding_ids:
                    deleted_count = delete_crow_embeddings(remove_embedding_ids)
                    logger.info(f"Deleted {deleted_count} embeddings for non-crow images")
                    success_count += len(remove_embedding_ids)
                else:
                    logger.warning("No embeddings found for images marked as not-crow")
            
            # Handle images to reassign (different crow)
            if images_to_reassign:
                logger.info(f"Processing {len(images_to_reassign)} images to reassign")
                
                # Get embedding IDs for images to reassign
                reassign_embedding_map = get_embedding_ids_by_image_paths(images_to_reassign)
                reassign_embedding_ids = list(reassign_embedding_map.values())
                
                if reassign_embedding_ids:
                    # Ask user what to do with reassigned images
                    choice = self.ask_reassignment_choice(len(reassign_embedding_ids))
                    
                    if choice == "new_crow":
                        # Create a new crow for these embeddings
                        new_crow_name = f"Split from Crow{self.current_crow_id}"
                        new_crow_id = create_new_crow_from_embeddings(
                            self.current_crow_id, 
                            reassign_embedding_ids, 
                            new_crow_name
                        )
                        logger.info(f"Created new crow {new_crow_id} with {len(reassign_embedding_ids)} embeddings")
                        success_count += len(reassign_embedding_ids)
                        
                    elif choice == "existing_crow":
                        # Let user select an existing crow
                        target_crow_id = self.select_target_crow()
                        if target_crow_id:
                            moved_count = reassign_crow_embeddings(
                                self.current_crow_id,
                                target_crow_id,
                                reassign_embedding_ids
                            )
                            logger.info(f"Moved {moved_count} embeddings to crow {target_crow_id}")
                            success_count += moved_count
                        else:
                            logger.info("User cancelled crow selection")
                    else:
                        logger.info("User cancelled reassignment")
                else:
                    logger.warning("No embeddings found for images marked for reassignment")
            
            # Handle images marked as multi-crow (remove from current crow's embeddings and add appropriate label)
            if images_multi_crow:
                logger.info(f"Processing {len(images_multi_crow)} images marked as multi-crow")
                
                # Get embedding IDs for images marked as multi-crow
                multi_crow_embedding_map = get_embedding_ids_by_image_paths(images_multi_crow)
                multi_crow_embedding_ids = list(multi_crow_embedding_map.values())
                
                if multi_crow_embedding_ids:
                    deleted_count = delete_crow_embeddings(multi_crow_embedding_ids)
                    logger.info(f"Deleted {deleted_count} embeddings for multi-crow images")
                    success_count += len(multi_crow_embedding_ids)
                else:
                    logger.warning("No embeddings found for images marked as multi-crow")
            
            return success_count == total_operations
            
        except Exception as e:
            logger.error(f"Error processing reassignments: {e}")
            return False
            
    def ask_reassignment_choice(self, num_images):
        """Ask user how to handle reassigned images."""
        dialog = tk.Toplevel(self.root)
        dialog.title("Reassignment Options")
        dialog.geometry("400x200")
        dialog.transient(self.root)
        dialog.grab_set()
        
        result = {"choice": None}
        
        # Center the dialog
        dialog.update_idletasks()
        x = self.root.winfo_x() + (self.root.winfo_width() - dialog.winfo_width()) // 2
        y = self.root.winfo_y() + (self.root.winfo_height() - dialog.winfo_height()) // 2
        dialog.geometry(f"+{x}+{y}")
        
        # Content
        frame = ttk.Frame(dialog, padding="20")
        frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(frame, text=f"You have {num_images} images marked as 'different crow'.", 
                 font=("Arial", 10, "bold")).pack(pady=(0, 10))
        
        ttk.Label(frame, text="How would you like to handle these images?").pack(pady=(0, 15))
        
        def on_new_crow():
            result["choice"] = "new_crow"
            dialog.destroy()
            
        def on_existing_crow():
            result["choice"] = "existing_crow"
            dialog.destroy()
            
        def on_cancel():
            result["choice"] = "cancel"
            dialog.destroy()
        
        ttk.Button(frame, text="Create New Crow", command=on_new_crow).pack(pady=2, fill=tk.X)
        ttk.Button(frame, text="Move to Existing Crow", command=on_existing_crow).pack(pady=2, fill=tk.X)
        ttk.Button(frame, text="Cancel", command=on_cancel).pack(pady=(10, 0), fill=tk.X)
        
        self.root.wait_window(dialog)
        return result["choice"]
        
    def select_target_crow(self):
        """Let user select a target crow for reassignment."""
        dialog = tk.Toplevel(self.root)
        dialog.title("Select Target Crow")
        dialog.geometry("300x400")
        dialog.transient(self.root)
        dialog.grab_set()
        
        result = {"crow_id": None}
        
        # Center the dialog
        dialog.update_idletasks()
        x = self.root.winfo_x() + (self.root.winfo_width() - dialog.winfo_width()) // 2
        y = self.root.winfo_y() + (self.root.winfo_height() - dialog.winfo_height()) // 2
        dialog.geometry(f"+{x}+{y}")
        
        # Content
        frame = ttk.Frame(dialog, padding="20")
        frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(frame, text="Select the crow to move images to:", 
                 font=("Arial", 10, "bold")).pack(pady=(0, 10))
        
        # Crow listbox
        listbox_frame = ttk.Frame(frame)
        listbox_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        crow_listbox = tk.Listbox(listbox_frame)
        crow_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(listbox_frame, orient="vertical", command=crow_listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        crow_listbox.configure(yscrollcommand=scrollbar.set)
        
        # Populate with crows (excluding current crow)
        try:
            crows = get_all_crows()
            crow_data = []
            for crow in crows:
                if crow['id'] != self.current_crow_id:  # Exclude current crow
                    name = crow['name'] if crow['name'] else f"Crow{crow['id']}"
                    display_text = f"Crow{crow['id']} ({name}) - {crow['total_sightings']} sightings"
                    crow_listbox.insert(tk.END, display_text)
                    crow_data.append(crow['id'])
        except Exception as e:
            logger.error(f"Error loading crows: {e}")
            messagebox.showerror("Error", "Failed to load crow list")
            dialog.destroy()
            return None
        
        def on_select():
            selection = crow_listbox.curselection()
            if selection:
                result["crow_id"] = crow_data[selection[0]]
                dialog.destroy()
            else:
                messagebox.showwarning("Warning", "Please select a crow")
                
        def on_cancel():
            dialog.destroy()
        
        button_frame = ttk.Frame(frame)
        button_frame.pack(fill=tk.X)
        
        ttk.Button(button_frame, text="Select", command=on_select).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(button_frame, text="Cancel", command=on_cancel).pack(side=tk.LEFT)
        
        self.root.wait_window(dialog)
        return result["crow_id"]
            
    def discard_changes(self):
        """Discard unsaved changes."""
        if self.unsaved_changes:
            result = messagebox.askyesno("Confirm Discard", 
                                       "Are you sure you want to discard all unsaved changes?")
            if result:
                # Reset name field
                self.name_var.set(self.current_crow_name or "")
                
                # Reset image selections
                for var in self.image_selections.values():
                    var.set("same_crow")
                    
                self.unsaved_changes = False
                self.save_button.configure(state='disabled')
                self.discard_button.configure(state='disabled')

def main():
    """Main function to run the suspect lineup tool."""
    root = tk.Tk()
    app = SuspectLineup(root)
    root.mainloop()

if __name__ == "__main__":
    main() 