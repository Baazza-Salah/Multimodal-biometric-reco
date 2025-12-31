# Fingerprint Biometric Identification Web App
# Flask backend for fingerprint registration and identification

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

from src.core.extractor import FingerprintFeatureExtractor
from src.core.matchers.math_matcher import MathMatcher, load_database
from src.core.matchers.ml_matcher import MLMatcher
from scripts.train import train_model as train_ml_model

app = Flask(__name__)

# Configuration
DATA_DIR = PROJECT_ROOT / 'data'
CSV_PATH = DATA_DIR / 'features.csv'
ML_MODEL_PATH = PROJECT_ROOT / 'models' / 'trained_model.pkl'

# Initialize extractor
extractor = FingerprintFeatureExtractor()

def save_upload(file, filename):
    """Save uploaded file to temp path."""
    temp_path = SCRIPT_DIR / filename
    file.save(temp_path)
    return temp_path

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/extract_feature', methods=['POST'])
def extract_feature():
    """Extract features from uploaded fingerprint image."""
    try:
        if 'file' not in request.files:
            return jsonify({'status': 'error', 'message': 'No file uploaded'})
            
        file = request.files['file']
        name = request.form.get('name')
        
        if not name or file.filename == '':
            return jsonify({'status': 'error', 'message': 'Name and file are required'})

        # Save temporary image
        temp_path = save_upload(file, 'temp_register.bmp')
        
        # Extract features
        try:
            features = extractor.extract_features(temp_path)
            
            # Add filename with name (e.g., user1_1.bmp)
            # We need to handle unique filenames or just use what user provided + timestamp or count
            # For simplicity, let's assume user handles unique naming or we append a count if we had state
            # But here we just use the provided name and maybe original extension
            ext = Path(file.filename).suffix
            filename = f"{name}{ext}" 
            features['filename'] = filename
            
            # Append to CSV
            df = pd.DataFrame([features])
            
            if not CSV_PATH.exists():
                df.to_csv(CSV_PATH, index=False)
            else:
                df.to_csv(CSV_PATH, mode='a', header=False, index=False)
                
            return jsonify({'status': 'success', 'message': f'Features extracted for {name}'})
            
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
    """Identify person from uploaded fingerprint."""
    try:
        if 'file' not in request.files:
            return jsonify({'status': 'error', 'message': 'No file uploaded'})
            
        file = request.files['file']
        if file.filename == '':
            return jsonify({'status': 'error', 'message': 'No file selected'})
            
        temp_path = save_upload(file, 'temp_probe.bmp')
        
        # Extract features
        features_dict = extractor.extract_features(temp_path)
        del features_dict['filename']
        probe_features = np.array(list(features_dict.values()), dtype='float32')
        
        results = {}
        
        # 1. Math Matcher Identification
        if CSV_PATH.exists():
            labels, db_features = load_database(CSV_PATH)
            if len(labels) > 0:
                math_matcher = MathMatcher()
                math_matcher.fit(db_features, labels)
                math_id, math_conf, _ = math_matcher.identify(probe_features, min_absolute=70.0)
                
                results['math'] = {
                    'identified': math_id if math_conf['absolute'] >= 70.0 else "Unknown",
                    'confidence': math_conf['absolute'],
                    'gap': math_conf['gap']
                }
            else:
                 results['math'] = {'error': 'Database empty'}
        else:
            results['math'] = {'error': 'Database not found'}
        
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
    app.run(host='0.0.0.0', port=1235, debug=True)
