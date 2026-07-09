@echo off
REM Setup script for OCR system (Windows)
REM Usage: setup.bat

echo.
echo ========================================================
echo   Image-to-Text Extraction Module - Setup Script
echo ========================================================
echo.

REM Check Python version
echo [1/5] Checking Python version...
python --version
if errorlevel 1 (
    echo  Error: Python not found. Please install Python 3.8+
    pause
    exit /b 1
)

REM Create virtual environment
echo.
echo [2/5] Creating virtual environment...
if not exist "venv" (
    python -m venv venv
    echo   Virtual environment created
) else (
    echo   Virtual environment already exists
)

REM Activate virtual environment
echo.
echo [3/5] Activating virtual environment...
call venv\Scripts\activate.bat
echo   Virtual environment activated

REM Install dependencies
echo.
echo [4/5] Installing dependencies...
python -m pip install --upgrade pip
pip install -r requirements.txt
if errorlevel 1 (
    echo   Error: Failed to install dependencies
    pause
    exit /b 1
)
echo   Dependencies installed

REM Create directories
echo.
echo [5/5] Creating project directories...
if not exist "models" mkdir models
if not exist "data" mkdir data
if not exist "logs" mkdir logs
if not exist "results" mkdir results
if not exist "models\checkpoints" mkdir models\checkpoints
echo   Directories created

echo.
echo ========================================================
echo   Setup Complete!
echo ========================================================
echo.
echo Next steps:
echo   1. Activate environment: venv\Scripts\activate.bat
echo   2. Run examples: python examples.py
echo   3. Run quickstart: python quickstart.py
echo   4. Run tests: python -m tests.test_ocr
echo.
pause
