"""
Flask web application for Image-to-Text Extraction.
Simple REST API with HTML frontend for extracting text from images.

Run with: python flask_app.py
Access at: http://localhost:5000
"""

from flask import Flask, render_template, request, jsonify, send_file
import numpy as np
import cv2
from PIL import Image
import io
import os
import tempfile
import json
from werkzeug.utils import secure_filename
from datetime import datetime

from src.preprocessing import ImagePreprocessor
from src.ocr_extractor import FakeNewsTextExtractor
from src.decoder import TextMetrics
from src.demo_ocr import DemoOCRExtractor, DemoFakeNewsExtractor
from src.inference_real import RealOCRInference
from config.config import IMAGE_WIDTH, IMAGE_HEIGHT

# Flask app setup
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024  # 200MB max
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Initialize OCR
ocr_model = None
fake_news_extractor = None
preprocessor = ImagePreprocessor()

def load_models():
    """Load OCR models on startup."""
    global ocr_model, fake_news_extractor
    model_path = os.path.join(BASE_DIR, 'models', 'best_ocr_model.h5')

    if not os.path.exists(model_path):
        ocr_model = DemoOCRExtractor()
        fake_news_extractor = DemoFakeNewsExtractor()
        print("✓ Demo mode activated (no trained model found)")
        print("💡 Tip: Train a model with: python train_ocr_simple.py")
        return

    try:
        ocr_model = RealOCRInference(model_path)
        print(f"✓ Real model loaded from {model_path}")
        print("✓ Production mode activated")
    except Exception as e:
        print(f"✗ Error loading real OCR model: {e}")
        print("✓ Falling back to demo mode")
        ocr_model = DemoOCRExtractor()
        fake_news_extractor = DemoFakeNewsExtractor()
        return

    try:
        fake_news_extractor = FakeNewsTextExtractor(model_path)
        print("✓ Fake news analyzer loaded")
    except Exception as e:
        fake_news_extractor = None
        print(f"⚠ Fake news analyzer unavailable: {e}")

@app.route('/')
def index():
    """Serve the main page."""
    return render_template('index.html', model_available=ocr_model is not None)

@app.route('/api/extract', methods=['POST'])
def extract_text():
    """Extract text from uploaded image."""
    try:
        # Check if file was uploaded
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Get options
        use_fake_news = request.form.get('fake_news', 'false').lower() == 'true'
        beam_width = int(request.form.get('beam_width', 50))
        confidence_threshold = float(request.form.get('confidence', 0.5))
        
        # Read image
        img = Image.open(file.stream)
        img_array = np.array(img)
        
        # Convert RGB to BGR if needed
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        result = {
            'image_info': {
                'width': img.width,
                'height': img.height,
                'format': img.format,
                'mode': img.mode
            },
            'timestamp': datetime.now().isoformat()
        }
        
        # Extract text
        if ocr_model:
            # Check if it's demo mode
            is_demo = isinstance(ocr_model, DemoOCRExtractor)
            
            if is_demo:
                # Demo mode - extract without saving file
                ocr_result = ocr_model.extract_text('')
            else:
                # Real model - save temp file and extract
                temp_path = os.path.join(app.config['UPLOAD_FOLDER'], 'temp_ocr.jpg')
                cv2.imwrite(temp_path, img_array)
                
                try:
                    ocr_result = ocr_model.extract_text(temp_path)
                except Exception as e:
                    print(f"Error during text extraction: {e}")
                    ocr_result = {'extracted_text': '(Extraction error)', 'confidence': 0}
                finally:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
            
            result['extracted_text'] = ocr_result.get('extracted_text', '')
            result['confidence'] = float(ocr_result.get('confidence', 0))
            result['text_length'] = len(result['extracted_text'])
            result['mode'] = 'demo' if is_demo else 'production'
            
            # Fake news analysis if requested
            if use_fake_news and fake_news_extractor:
                try:
                    if isinstance(fake_news_extractor, DemoFakeNewsExtractor):
                        # Demo analysis
                        analysis_result = fake_news_extractor.extract_for_analysis('')
                    else:
                        # Real analysis
                        analysis_result = {
                            'risk_level': 'low',
                            'warnings': [],
                            'features': fake_news_extractor._extract_text_features(
                                result['extracted_text']
                            ),
                            'readability': fake_news_extractor._calculate_readability(
                                result['extracted_text']
                            )
                        }
                        analysis_result['flags'] = fake_news_extractor._detect_suspicious_patterns(
                            result['extracted_text']
                        )
                    
                    result['fake_news_analysis'] = {
                        'risk_level': analysis_result.get('risk_level', 'low'),
                        'warnings': analysis_result.get('warnings', []),
                        'features': analysis_result.get('features', {}),
                        'readability': analysis_result.get('readability', {}),
                        'flags': analysis_result.get('flags', {})
                    }
                except Exception as e:
                    result['fake_news_warning'] = f"Analysis error: {str(e)}"
            elif use_fake_news and not fake_news_extractor:
                result['fake_news_warning'] = 'Fake news analysis is unavailable in current mode.'
        else:
            # No model at all - shouldn't happen with fallback
            result['extracted_text'] = '(Error: No extraction mode available)'
            result['confidence'] = None
            result['text_length'] = 0
        
        result['success'] = True
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/api/preprocess', methods=['POST'])
def preprocess():
    """Preprocess image and return statistics."""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        img = Image.open(file.stream)
        img_array = np.array(img)
        
        # Convert to BGR if needed
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Preprocess
        processed = preprocessor.preprocess(img_array)
        
        return jsonify({
            'success': True,
            'input_shape': img_array.shape,
            'output_shape': processed.shape,
            'value_range': [float(processed.min()), float(processed.max())],
            'mean': float(processed.mean()),
            'std': float(processed.std())
        })
    
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint."""
    is_demo = isinstance(ocr_model, DemoOCRExtractor)
    is_real = isinstance(ocr_model, RealOCRInference)
    return jsonify({
        'status': 'healthy',
        'model_available': ocr_model is not None,
        'mode': 'demo' if is_demo else ('production' if is_real else 'unknown'),
        'model_type': 'real' if is_real else ('demo' if is_demo else 'none'),
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    print("=" * 60)
    print("Image-to-Text Extraction Web Server")
    print("=" * 60)
    
    # Load models
    load_models()
    
    print("\n🚀 Starting Flask server...")
    print("📍 Access at: http://localhost:5000")
    print("=" * 60)
    
    # Run Flask app
    app.run(debug=False, host='0.0.0.0', port=5000, use_reloader=False)
