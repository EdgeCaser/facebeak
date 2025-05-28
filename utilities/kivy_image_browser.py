#!/usr/bin/env python3
"""
Kivy Image Browser for SSH - Browse and view crop images remotely
Usage: python kivy_image_browser.py [starting_directory]
"""

import os
import sys
from pathlib import Path
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.scrollview import ScrollView
from kivy.uix.image import Image
from kivy.uix.popup import Popup
from kivy.uix.filechooser import FileChooserListView
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.core.window import Window
import threading
import queue

class ImageBrowser(BoxLayout):
    def __init__(self, start_path=None, **kwargs):
        super().__init__(**kwargs)
        self.orientation = 'horizontal'
        
        # Set starting directory
        if start_path and os.path.exists(start_path):
            self.current_path = start_path
        else:
            self.current_path = os.getcwd()
        
        # Image extensions to filter
        self.image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
        
        # Current image list and index
        self.image_files = []
        self.current_image_index = 0
        
        # Queue for thread-safe image loading
        self.image_queue = queue.Queue()
        
        self.build_ui()
        self.load_directory()
        
        # Schedule periodic check for loaded images
        Clock.schedule_interval(self.check_image_queue, 0.1)
    
    def build_ui(self):
        # Left panel - File browser
        left_panel = BoxLayout(orientation='vertical', size_hint=(0.3, 1))
        
        # Directory navigation
        nav_layout = BoxLayout(orientation='horizontal', size_hint=(1, 0.1))
        
        self.path_label = Label(
            text=f"Path: {self.current_path}", 
            text_size=(None, None),
            halign='left',
            size_hint=(0.7, 1)
        )
        nav_layout.add_widget(self.path_label)
        
        up_btn = Button(text="Up", size_hint=(0.15, 1))
        up_btn.bind(on_press=self.go_up_directory)
        nav_layout.add_widget(up_btn)
        
        refresh_btn = Button(text="Refresh", size_hint=(0.15, 1))
        refresh_btn.bind(on_press=self.refresh_directory)
        nav_layout.add_widget(refresh_btn)
        
        left_panel.add_widget(nav_layout)
        
        # File chooser
        self.file_chooser = FileChooserListView(
            path=self.current_path,
            size_hint=(1, 0.9)
        )
        self.file_chooser.bind(on_selection=self.on_file_selection)
        left_panel.add_widget(self.file_chooser)
        
        self.add_widget(left_panel)
        
        # Right panel - Image viewer
        right_panel = BoxLayout(orientation='vertical', size_hint=(0.7, 1))
        
        # Image info and controls
        info_layout = BoxLayout(orientation='horizontal', size_hint=(1, 0.1))
        
        self.image_info = Label(
            text="Select an image or directory",
            text_size=(None, None),
            halign='left'
        )
        info_layout.add_widget(self.image_info)
        
        # Navigation buttons
        prev_btn = Button(text="Previous", size_hint=(0.15, 1))
        prev_btn.bind(on_press=self.previous_image)
        info_layout.add_widget(prev_btn)
        
        next_btn = Button(text="Next", size_hint=(0.15, 1))
        next_btn.bind(on_press=self.next_image)
        info_layout.add_widget(next_btn)
        
        right_panel.add_widget(info_layout)
        
        # Image display
        self.image_widget = Image(
            size_hint=(1, 0.9),
            allow_stretch=True,
            keep_ratio=True
        )
        right_panel.add_widget(self.image_widget)
        
        self.add_widget(right_panel)
    
    def load_directory(self):
        """Load images from current directory"""
        try:
            all_files = os.listdir(self.current_path)
            self.image_files = []
            
            for file in sorted(all_files):
                file_path = os.path.join(self.current_path, file)
                if os.path.isfile(file_path):
                    _, ext = os.path.splitext(file.lower())
                    if ext in self.image_extensions:
                        self.image_files.append(file)
            
            self.current_image_index = 0
            self.update_image_info()
            
            if self.image_files:
                self.load_current_image()
            else:
                self.image_widget.source = ""
                
        except Exception as e:
            self.show_error(f"Error loading directory: {str(e)}")
    
    def update_image_info(self):
        """Update image information display"""
        if self.image_files:
            current_file = self.image_files[self.current_image_index]
            file_path = os.path.join(self.current_path, current_file)
            
            try:
                file_size = os.path.getsize(file_path)
                size_str = self.format_file_size(file_size)
                info_text = f"{self.current_image_index + 1}/{len(self.image_files)}: {current_file} ({size_str})"
            except:
                info_text = f"{self.current_image_index + 1}/{len(self.image_files)}: {current_file}"
                
            self.image_info.text = info_text
        else:
            self.image_info.text = f"No images in {self.current_path}"
    
    def format_file_size(self, size_bytes):
        """Format file size in human readable format"""
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.1f} KB"
        else:
            return f"{size_bytes / (1024 * 1024):.1f} MB"
    
    def load_current_image(self):
        """Load current image in background thread"""
        if not self.image_files:
            return
            
        current_file = self.image_files[self.current_image_index]
        file_path = os.path.join(self.current_path, current_file)
        
        # Load image in background thread
        threading.Thread(
            target=self.load_image_thread,
            args=(file_path,),
            daemon=True
        ).start()
    
    def load_image_thread(self, file_path):
        """Load image in background thread"""
        try:
            # Put the file path in queue for main thread to handle
            self.image_queue.put(('load', file_path))
        except Exception as e:
            self.image_queue.put(('error', str(e)))
    
    def check_image_queue(self, dt):
        """Check for images loaded in background thread"""
        try:
            while not self.image_queue.empty():
                action, data = self.image_queue.get_nowait()
                
                if action == 'load':
                    self.image_widget.source = data
                elif action == 'error':
                    self.show_error(f"Error loading image: {data}")
                    
        except queue.Empty:
            pass
    
    def on_file_selection(self, instance, selection):
        """Handle file/directory selection"""
        if not selection:
            return
            
        selected_path = selection[0]
        
        if os.path.isdir(selected_path):
            # Directory selected - navigate to it
            self.current_path = selected_path
            self.path_label.text = f"Path: {self.current_path}"
            self.file_chooser.path = self.current_path
            self.load_directory()
        elif os.path.isfile(selected_path):
            # File selected - show it if it's an image
            _, ext = os.path.splitext(selected_path.lower())
            if ext in self.image_extensions:
                filename = os.path.basename(selected_path)
                if filename in self.image_files:
                    self.current_image_index = self.image_files.index(filename)
                    self.update_image_info()
                    self.load_current_image()
    
    def go_up_directory(self, instance):
        """Navigate to parent directory"""
        parent = os.path.dirname(self.current_path)
        if parent != self.current_path:  # Not at root
            self.current_path = parent
            self.path_label.text = f"Path: {self.current_path}"
            self.file_chooser.path = self.current_path
            self.load_directory()
    
    def refresh_directory(self, instance):
        """Refresh current directory"""
        self.file_chooser.path = self.current_path
        self.load_directory()
    
    def previous_image(self, instance):
        """Show previous image"""
        if self.image_files and len(self.image_files) > 1:
            self.current_image_index = (self.current_image_index - 1) % len(self.image_files)
            self.update_image_info()
            self.load_current_image()
    
    def next_image(self, instance):
        """Show next image"""
        if self.image_files and len(self.image_files) > 1:
            self.current_image_index = (self.current_image_index + 1) % len(self.image_files)
            self.update_image_info()
            self.load_current_image()
    
    def show_error(self, message):
        """Show error popup"""
        popup = Popup(
            title='Error',
            content=Label(text=message),
            size_hint=(0.6, 0.4)
        )
        popup.open()

class ImageBrowserApp(App):
    def __init__(self, start_path=None, **kwargs):
        super().__init__(**kwargs)
        self.start_path = start_path
    
    def build(self):
        # Set window title
        self.title = "Kivy Image Browser"
        
        # Set minimum window size
        Window.minimum_width = 800
        Window.minimum_height = 600
        
        return ImageBrowser(start_path=self.start_path)

def main():
    """Main entry point"""
    start_path = None
    
    if len(sys.argv) > 1:
        start_path = sys.argv[1]
        if not os.path.exists(start_path):
            print(f"Warning: Path '{start_path}' does not exist. Using current directory.")
            start_path = None
    
    # Default to crow_crops directory if it exists
    if not start_path:
        possible_paths = [
            'crow_crops',
            '../crow_crops',
            '/home/ubuntu/facebeak/crow_crops',
            os.path.expanduser('~/facebeak/crow_crops')
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                start_path = path
                break
    
    print(f"Starting Image Browser...")
    if start_path:
        print(f"Starting directory: {start_path}")
    else:
        print(f"Starting directory: {os.getcwd()}")
    
    app = ImageBrowserApp(start_path=start_path)
    app.run()

if __name__ == '__main__':
    main() 