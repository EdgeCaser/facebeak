#!/usr/bin/env python3
"""
Facebeak GUI Launcher
Ensures the GUI runs with proper environment setup.
"""

import sys
import os
import subprocess
from pathlib import Path

def check_venv():
    """Check if virtual environment exists and is activated."""
    venv_path = Path("venv")
    if not venv_path.exists():
        print("❌ Virtual environment not found!")
        print("Run: python setup_environment.py")
        return False
    
    # Check if we're in venv
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("✅ Virtual environment is active")
        return True
    else:
        print("⚠️  Virtual environment not active")
        return False

def run_gui():
    """Run the Facebeak GUI."""
    if check_venv():
        # Already in venv, run directly
        print("🚀 Starting Facebeak GUI...")
        os.system("python kivy_extract_training_gui.py")
    else:
        # Activate venv and run
        print("🔄 Activating virtual environment and starting GUI...")
        if os.name == 'nt':  # Windows
            cmd = "venv\\Scripts\\python.exe kivy_extract_training_gui.py"
        else:  # Unix/Linux/macOS
            cmd = "venv/bin/python kivy_extract_training_gui.py"
        
        os.system(cmd)

if __name__ == "__main__":
    print("🦅 Facebeak GUI Launcher")
    print("=" * 30)
    run_gui() 