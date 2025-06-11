#!/usr/bin/env python3
"""
Launcher script for the Crow Image Ingestion Tool.
This tool allows you to ingest images, detect crows, extract 512x512 crops,
and optionally label them during the process.
"""

import sys
import os

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from gui.image_ingestion_gui import main

if __name__ == "__main__":
    main() 