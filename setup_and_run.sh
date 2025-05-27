#!/usr/bin/env bash

# Facebeak Development Environment Setup (Shell Script)
# This script activates the virtual environment, installs dependencies,
# and leaves you in an activated environment.

echo "========================================"
echo "Facebeak Development Environment Setup"
echo "========================================"
echo ""

# Get the directory where this script is located
# and change to it to ensure paths are relative to project root.
PROJECT_ROOT=$(dirname "$0")
cd "$PROJECT_ROOT" || { echo "Error: Could not change to project root directory. Exiting."; exit 1; }

echo "Current directory: $(pwd)"
echo ""

# Define the path to the virtual environment activation script
# Common names are .venv or venv. Adjust if yours is different.
VENV_DIR=".venv" # Using .venv as per task description
VENV_ACTIVATE_1="$VENV_DIR/bin/activate"
VENV_ACTIVATE_2="venv/bin/activate" # Common alternative

# Check if virtual environment exists
if [ ! -f "$VENV_ACTIVATE_1" ] && [ ! -f "$VENV_ACTIVATE_2" ]; then
  echo "Error: Virtual environment not found."
  echo "Please create a virtual environment first. For example, in the project root:"
  echo "  python3 -m venv $VENV_DIR"
  echo "Or, if you use 'python':"
  echo "  python -m venv $VENV_DIR"
  echo ""
  exit 1
fi

echo "Activating virtual environment..."
if [ -f "$VENV_ACTIVATE_1" ]; then
  source "$VENV_ACTIVATE_1"
  echo "Activated from $VENV_ACTIVATE_1"
elif [ -f "$VENV_ACTIVATE_2" ]; then
  source "$VENV_ACTIVATE_2"
  echo "Activated from $VENV_ACTIVATE_2"
else
  # This case should ideally not be reached due to the check above,
  # but as a safeguard:
  echo "Error: Failed to activate virtual environment. Activation script not found."
  exit 1
fi

# Check if activation was successful (e.g., by checking if VIRTUAL_ENV is set)
if [ -z "$VIRTUAL_ENV" ]; then
  echo "Error: Failed to activate virtual environment."
  echo "Make sure your virtual environment is correctly set up."
  exit 1
fi

echo "Virtual environment activated successfully!"
echo "Virtual environment path: $VIRTUAL_ENV"
echo ""

# Upgrade pip first
echo "Upgrading pip..."
python -m pip install --upgrade pip

if [ $? -ne 0 ]; then
    echo "Warning: Failed to upgrade pip, continuing anyway..."
fi

echo ""

# Install dependencies
echo "Installing dependencies from requirements.txt..."
if [ -f "requirements.txt" ]; then
  pip install -r requirements.txt
  if [ $? -ne 0 ]; then
      echo "Error: Failed to install some dependencies."
      echo "You may need to install them manually or check your internet connection."
      echo ""
      echo "Common issues:"
      echo "- For GPU support, you may need to install PyTorch with CUDA separately."
      echo "  Visit https://pytorch.org/ for instructions."
      echo ""
  fi
else
  echo "Warning: requirements.txt not found. Skipping dependency installation."
fi

echo ""
echo "========================================"
echo "Setup Complete!"
echo "========================================"
echo ""
echo "Virtual environment is activated and dependencies are (attempted to be) installed."
echo "You are now in the project root directory: $(pwd)"
echo ""
echo "Available commands (examples):"
echo "  python main.py --help                    - Run main facebeak application with CLI arguments"
echo "  python facebeak.py                       - Run the Facebeak GUI launcher"
echo "  python utilities/extract_training_gui.py - Run training data extraction GUI"
echo "  python -m pytest                        - Run tests"
echo "  deactivate                              - Exit virtual environment"
echo ""
echo "The virtual environment will remain active in this terminal session."
echo "To exit the virtual environment, type 'deactivate'."
echo "To close this terminal, simply type 'exit' or close the window."
echo ""
