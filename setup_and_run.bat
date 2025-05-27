@echo off
REM Facebeak Development Environment Setup
REM This script activates the virtual environment, installs dependencies, and opens a CLI

echo ========================================
echo Facebeak Development Environment Setup
echo ========================================
echo.

REM Get the directory where this batch file is located
set PROJECT_ROOT=%~dp0
cd /d "%PROJECT_ROOT%"

echo Current directory: %CD%
echo.

REM Check if virtual environment exists
if not exist ".venv\Scripts\activate.bat" (
    echo Virtual environment not found at .venv\Scripts\activate.bat
    echo Please create a virtual environment first:
    echo   python -m venv .venv
    echo.
    pause
    exit /b 1
)

echo Activating virtual environment...
call .venv\Scripts\activate.bat

if errorlevel 1 (
    echo Failed to activate virtual environment
    pause
    exit /b 1
)

echo Virtual environment activated successfully!
echo.

REM Upgrade pip first
echo Upgrading pip...
python -m pip install --upgrade pip

if errorlevel 1 (
    echo Warning: Failed to upgrade pip, continuing anyway...
)

echo.

REM Install dependencies
echo Installing dependencies from requirements.txt...
pip install -r requirements.txt

if errorlevel 1 (
    echo Error: Failed to install some dependencies
    echo You may need to install them manually or check your internet connection
    echo.
    echo Common issues:
    echo - For GPU support, you may need to install PyTorch with CUDA:
    echo   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    echo.
    pause
)

echo.
echo ========================================
echo Setup Complete!
echo ========================================
echo.
echo Virtual environment is activated and dependencies are installed.
echo You are now in the project root directory: %CD%
echo.
echo Available commands:
echo   python main.py --help                    - Run main facebeak application
echo   python utilities/extract_training_gui.py - Run training data extraction GUI
echo   python facebeak.py --help               - Run facebeak CLI
echo   python -m pytest                        - Run tests
echo   deactivate                              - Exit virtual environment
echo.
echo Type 'exit' to close this window.
echo.

REM Keep the command prompt open with the virtual environment activated
cmd /k 