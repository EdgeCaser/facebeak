#!/usr/bin/env bash

# Script to run facebeak with user input

# Define the path to the virtual environment activation script
# Common names are .venv or venv. Adjust if yours is different.
VENV_PATH_1=".venv/bin/activate"
VENV_PATH_2="venv/bin/activate"

# Activate virtual environment
if [ -f "$VENV_PATH_1" ]; then
  echo "Activating virtual environment from $VENV_PATH_1..."
  source "$VENV_PATH_1"
elif [ -f "$VENV_PATH_2" ]; then
  echo "Activating virtual environment from $VENV_PATH_2..."
  source "$VENV_PATH_2"
else
  echo "Error: Virtual environment not found at $VENV_PATH_1 or $VENV_PATH_2."
  echo "Please create and activate your virtual environment first."
  exit 1
fi

# Change to the repository root directory (optional, if script is not already there)
# REPO_PATH=$(dirname "$0")
# cd "$REPO_PATH" || exit

echo ""
echo "Facebeak Video Processor"
echo "------------------------"

# Get user input
read -p "Enter path to input video (e.g., sample.mp4): " VIDEO_PATH
read -p "Enter path for frame-skipped output video (e.g., skip_output.mp4): " SKIP_OUTPUT_PATH
read -p "Enter path for full-frame interpolated output video (e.g., full_output.mp4): " FULL_OUTPUT_PATH
read -p "Enter detection threshold (default 0.3): " DET_THRESH
read -p "Enter frame skip for processing (default 5): " FRAME_SKIP
read -p "Preserve audio? (yes/no, default yes): " PRESERVE_AUDIO_INPUT

# Set default values if input is empty
VIDEO_PATH=${VIDEO_PATH:-sample.mp4}
SKIP_OUTPUT_PATH=${SKIP_OUTPUT_PATH:-skip_output.mp4}
FULL_OUTPUT_PATH=${FULL_OUTPUT_PATH:-full_output.mp4}
DET_THRESH=${DET_THRESH:-0.3}
FRAME_SKIP=${FRAME_SKIP:-5}
PRESERVE_AUDIO_INPUT=${PRESERVE_AUDIO_INPUT:-yes}

# Construct the command
CMD_ARGS=("--video" "$VIDEO_PATH" "--skip-output" "$SKIP_OUTPUT_PATH" "--full-output" "$FULL_OUTPUT_PATH" "--detection-threshold" "$DET_THRESH" "--skip" "$FRAME_SKIP")

if [[ "$PRESERVE_AUDIO_INPUT" == "yes" || "$PRESERVE_AUDIO_INPUT" == "y" ]]; then
  CMD_ARGS+=("--preserve-audio")
fi

echo ""
echo "Running facebeak with the following command:"
# Correctly quote each argument for display
CMD_DISPLAY="python main.py"
for arg in "${CMD_ARGS[@]}"; do
  CMD_DISPLAY+=" \"$arg\""
done
echo "$CMD_DISPLAY"
echo ""

# Execute the command
python main.py "${CMD_ARGS[@]}"

echo ""
echo "Processing finished."
# Deactivate virtual environment (optional)
# deactivate
read -p "Press Enter to exit..."
