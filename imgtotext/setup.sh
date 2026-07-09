#!/bin/bash

# Setup script for OCR system
# Usage: bash setup.sh

echo "╔═════════════════════════════════════════════════════════════╗"
echo "║   Image-to-Text Extraction Module - Setup Script            ║"
echo "╚═════════════════════════════════════════════════════════════╝"
echo ""

# Check Python version
echo "[1/5] Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "  Python version: $python_version"

# Create virtual environment
echo ""
echo "[2/5] Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "  Virtual environment created"
else
    echo "  Virtual environment already exists"
fi

# Activate virtual environment
echo ""
echo "[3/5] Activating virtual environment..."
source venv/bin/activate
echo "  Virtual environment activated"

# Install dependencies
echo ""
echo "[4/5] Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt
echo "  Dependencies installed"

# Create necessary directories
echo ""
echo "[5/5] Creating project directories..."
mkdir -p models
mkdir -p data
mkdir -p logs
mkdir -p results
mkdir -p models/checkpoints
echo "  Directories created"

echo ""
echo "╔═════════════════════════════════════════════════════════════╗"
echo "║   ✓ Setup Complete!                                         ║"
echo "╚═════════════════════════════════════════════════════════════╝"
echo ""
echo "Next steps:"
echo "  1. Activate environment: source venv/bin/activate"
echo "  2. Run examples: python examples.py"
echo "  3. Run quickstart: python quickstart.py"
echo "  4. Run tests: python -m tests.test_ocr"
echo ""
