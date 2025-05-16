@echo off
REM Batch script to run facebeak with one click

REM Set these paths as needed
set VENV_PATH=C:\Users\ianfe\venv\Scripts\activate
set REPO_PATH=C:\Users\ianfe\OneDrive\Documents\GitHub\birbTagger

call %VENV_PATH%
cd /d %REPO_PATH%

set /p VIDEO="Enter path to input video (e.g., sample.mp4): "
set /p OUTPUT="Enter path to output video (e.g., output.mp4): "
set /p DET="Enter detection threshold (default 0.3): "
set /p SKIP="Enter frame skip (default 1): "

if "%DET%"=="" set DET=0.3
if "%SKIP%"=="" set SKIP=1

python main.py --video %VIDEO% --output %OUTPUT% --detection-threshold %DET% --skip %SKIP%
pause 