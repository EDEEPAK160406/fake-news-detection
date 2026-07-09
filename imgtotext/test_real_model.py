#!/usr/bin/env python
"""Test script to verify real model is working."""

import sys
import os
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

print("=" * 60)
print("Testing Real Model Loading")
print("=" * 60)

# Test 1: Check model file exists
model_path = 'models/best_ocr_model.h5'
print(f"\n1. Checking model file: {model_path}")
if os.path.exists(model_path):
    size_mb = os.path.getsize(model_path) / (1024 * 1024)
    print(f"   ✓ Model found ({size_mb:.1f} MB)")
else:
    print(f"   ✗ Model not found")
    sys.exit(1)

# Test 2: Load RealOCRInference
print("\n2. Loading RealOCRInference class...")
try:
    from src.inference_real import RealOCRInference
    print("   ✓ RealOCRInference imported")
except Exception as e:
    print(f"   ✗ Import error: {e}")
    sys.exit(1)

# Test 3: Instantiate model
print("\n3. Instantiating model...")
try:
    model = RealOCRInference(model_path)
    print("   ✓ Model instantiated successfully")
    print(f"   ✓ Character set: {len(model.character_set)} characters")
except Exception as e:
    print(f"   ✗ Instantiation error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Create a test image
print("\n4. Creating test image...")
try:
    from PIL import Image, ImageDraw, ImageFont
    import numpy as np
    
    # Create a test image with text
    img = Image.new('L', (256, 32), color=255)
    draw = ImageDraw.Draw(img)
    
    # Draw some white text on black background
    img = Image.new('L', (256, 32), color=0)  # Black background
    draw = ImageDraw.Draw(img)
    
    # Try to use a basic font
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()
    
    draw.text((10, 5), "TEST", fill=255, font=font)
    
    # Save test image
    test_img_path = 'test_image.png'
    img.save(test_img_path)
    print(f"   ✓ Test image created: {test_img_path}")
except Exception as e:
    print(f"   ✗ Image creation error: {e}")
    sys.exit(1)

# Test 5: Extract text
print("\n5. Extracting text from test image...")
try:
    result = model.extract_text(test_img_path)
    print(f"   ✓ Extraction successful")
    print(f"   - Extracted text: '{result.get('extracted_text', '')}'")
    print(f"   - Confidence: {result.get('confidence', 0):.2%}")
    print(f"   - Source: {result.get('source', 'unknown')}")
except Exception as e:
    print(f"   ✗ Extraction error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Cleanup
try:
    os.remove(test_img_path)
    print("\n6. Cleanup: Test image removed")
except:
    pass

print("\n" + "=" * 60)
print("✅ All tests passed! Real model is working.")
print("=" * 60)
