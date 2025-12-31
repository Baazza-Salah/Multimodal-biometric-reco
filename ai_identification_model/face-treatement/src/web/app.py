# Biometric Identification Web App
# Flask backend for face registration and identification

import os
import sys
import cv2
import numpy as np
import base64
import pandas as pd
from pathlib import Path
from flask import Flask, render_template, request, jsonify

# Add parent directories to path for imports
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.core.extractor import FaceFeatureExtractor, MODEL_PATH as FACE_MODEL_PATH
from src.core.matchers.math_matcher import MathMatcher, load_database
from src.core.matchers.ml_matcher import MLMatcher
from scripts.train import train_model as train_ml_model

app = Flask(__name__)

# Configuration
DATA_DIR = PROJECT_ROOT / 'data'
CSV_PATH = DATA_DIR / 'features.csv'
ML_MODEL_PATH = PROJECT_ROOT / 'models' / 'trained_model.pkl'

# Initialize extractor
extractor = FaceFeatureExtractor(FACE_MODEL_PATH)

def base64_to_image(base64_string):
    """Convert base64 string to numpy image."""
    if ',' in base64_string:
        base64_string = base64_string.split(',')[1]
    img_data = base64.b64decode(base64_string)
    nparr = np.frombuffer(img_data, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/extract_feature', methods=['POST'])
def extract_feature():
    """Extract features from a frame sent by client."""
    try:
        data = request.json
        image_data = data['image']
        name = data['name']
        count = data['count']
        
        # Convert image
        image = base64_to_image(image_data)
        
        # Save temporary image for extractor
        temp_path = SCRIPT_DIR / 'temp_capture.png'
        cv2.imwrite(str(temp_path), image)
        
        # Extract features
        try:
            features = extractor.extract_features(temp_path)
            
            # Add filename with count (e.g., salah1, salah2)
            filename = f"{name}{count}.png"
            features['filename'] = filename
            
            # Append to CSV
            df = pd.DataFrame([features])
            
            if not CSV_PATH.exists():
                df.to_csv(CSV_PATH, index=False)
            else:
                df.to_csv(CSV_PATH, mode='a', header=False, index=False)
                
            return jsonify({'status': 'success', 'message': f'Feature {count} extracted'})
            
        except Exception as e:
            return jsonify({'status': 'error', 'message': str(e)})
            
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/train', methods=['POST'])
def train():
    """Retrain the ML model."""
    try:
        train_ml_model()
        return jsonify({'status': 'success', 'message': 'Model retrained successfully'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/identify', methods=['POST'])
def identify():
    """Identify person from uploaded image or base64 string."""
    temp_path = SCRIPT_DIR / 'temp_probe.png'
    
    # Handle File Upload
    if 'file' in request.files:
        file = request.files['file']
        if file.filename == '':
            return jsonify({'status': 'error', 'message': 'No file selected'})
        file.save(temp_path)
        
    # Handle Base64 Image (Camera)
    elif request.is_json:
        data = request.json
        if 'image' not in data:
            return jsonify({'status': 'error', 'message': 'No image data'})
        image = base64_to_image(data['image'])
        cv2.imwrite(str(temp_path), image)
        
    else:
        return jsonify({'status': 'error', 'message': 'No image provided'})
    
    try:
        # Extract features
        features_dict = extractor.extract_features(temp_path)
        del features_dict['filename']
        probe_features = np.array(list(features_dict.values()), dtype='float32')
        
        results = {}
        
        # 1. Math Matcher Identification
        labels, db_features = load_database(CSV_PATH)
        math_matcher = MathMatcher()
        math_matcher.fit(db_features, labels)
        math_id, math_conf, _ = math_matcher.identify(probe_features, min_absolute=70.0)
        
        results['math'] = {
            'identified': math_id if math_conf['absolute'] >= 70.0 else "Unknown",
            'confidence': math_conf['absolute'],
            'gap': math_conf['gap']
        }
        
        # 2. ML Matcher Identification
        if ML_MODEL_PATH.exists():
            ml_matcher = MLMatcher()
            ml_matcher.load(ML_MODEL_PATH)
            ml_id, ml_conf, _ = ml_matcher.identify(probe_features, min_confidence=70.0)
            
            results['ml'] = {
                'identified': ml_id if ml_conf['confidence'] >= 70.0 else "Unknown",
                'confidence': ml_conf['confidence'],
                'probability': ml_conf['probability']
            }
        else:
            results['ml'] = {'error': 'Model not trained'}
            
        return jsonify({'status': 'success', 'results': results})
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=1234, debug=True)
