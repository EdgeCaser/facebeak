#!/usr/bin/env python3
"""
Facebeak Environment Setup Script
Automatically sets up the development environment with all required dependencies.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd, cwd=None):
    """Run a command and return the result."""
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error running command: {cmd}")
        print(f"Error output: {result.stderr}")
        return False
    print(f"Success: {result.stdout.strip()}")
    return True

def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major != 3 or version.minor < 8:
        print(f"❌ Python {version.major}.{version.minor} detected. Requires Python 3.8+")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True

def setup_virtual_environment():
    """Set up virtual environment if it doesn't exist."""
    venv_path = Path("venv")
    if venv_path.exists():
        print("✅ Virtual environment already exists")
        return True
    
    print("📦 Creating virtual environment...")
    if not run_command(f"{sys.executable} -m venv venv"):
        return False
    
    print("✅ Virtual environment created")
    return True

def install_dependencies():
    """Install all required dependencies."""
    print("📦 Installing dependencies from requirements.txt...")
    
    # Determine the pip path based on OS
    if os.name == 'nt':  # Windows
        pip_path = "venv\\Scripts\\pip"
        python_path = "venv\\Scripts\\python"
    else:  # Unix/Linux/macOS
        pip_path = "venv/bin/pip"
        python_path = "venv/bin/python"
    
    # Upgrade pip first
    if not run_command(f"{python_path} -m pip install --upgrade pip"):
        return False
    
    # Install PyTorch with CUDA support first
    print("🔥 Installing PyTorch with CUDA support...")
    pytorch_cmd = f"{pip_path} install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"
    if not run_command(pytorch_cmd):
        print("⚠️  CUDA PyTorch installation failed, trying CPU version...")
        if not run_command(f"{pip_path} install torch torchvision torchaudio"):
            return False
    
    # Install other requirements
    if not run_command(f"{pip_path} install -r requirements.txt"):
        return False
    
    print("✅ All dependencies installed successfully")
    return True

def verify_installation():
    """Verify that critical packages can be imported."""
    print("🔍 Verifying installation...")
    
    # Determine python path
    if os.name == 'nt':  # Windows
        python_path = "venv\\Scripts\\python"
    else:  # Unix/Linux/macOS
        python_path = "venv/bin/python"
    
    test_imports = [
        ("torch", "PyTorch"),
        ("cv2", "OpenCV"),
        ("kivy", "Kivy"),
        ("numpy", "NumPy"),
        ("cryptography", "Cryptography"),
        ("ultralytics", "Ultralytics"),
        ("sklearn", "Scikit-learn"),
        ("matplotlib", "Matplotlib"),
    ]
    
    for module, name in test_imports:
        cmd = f'{python_path} -c "import {module}; print(f\\"✅ {name}: {{getattr({module}, \'__version__\', \'OK\')}}\\")"'
        if not run_command(cmd):
            print(f"❌ {name} import failed")
            return False
    
    return True

def test_facebeak_imports():
    """Test Facebeak-specific imports."""
    print("🎯 Testing Facebeak imports...")
    
    if os.name == 'nt':  # Windows
        python_path = "venv\\Scripts\\python"
    else:
        python_path = "venv/bin/python"
    
    facebeak_tests = [
        "from logging_config import setup_logging",
        "from db_security import secure_database_connection", 
        "from models import CrowResNetEmbedder",
        "from tracking import load_faster_rcnn",
    ]
    
    for test in facebeak_tests:
        cmd = f'{python_path} -c "{test}; print(\\"✅ {test}\\")"'
        if not run_command(cmd):
            print(f"❌ {test} failed")
            return False
    
    print("✅ All Facebeak imports working")
    return True

def main():
    """Main setup function."""
    print("🦅 Facebeak Environment Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Setup virtual environment
    if not setup_virtual_environment():
        print("❌ Failed to set up virtual environment")
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        print("❌ Failed to install dependencies")
        sys.exit(1)
    
    # Verify installation
    if not verify_installation():
        print("❌ Installation verification failed")
        sys.exit(1)
    
    # Test Facebeak imports
    if not test_facebeak_imports():
        print("❌ Facebeak import tests failed")
        sys.exit(1)
    
    print("\n" + "=" * 50)
    print("🎉 Environment setup complete!")
    print("\nTo activate the environment:")
    if os.name == 'nt':  # Windows
        print("  .\\venv\\Scripts\\Activate.ps1")
        print("\nTo run the GUI:")
        print("  python kivy_extract_training_gui.py")
    else:  # Unix/Linux/macOS
        print("  source venv/bin/activate")
        print("\nTo run the GUI:")
        print("  python kivy_extract_training_gui.py")

if __name__ == "__main__":
    main() 