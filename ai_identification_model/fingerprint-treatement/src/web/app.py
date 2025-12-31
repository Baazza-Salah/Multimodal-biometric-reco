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
            # Returns a list of dicts (original + augmented)
            features_list = extractor.extract_features(temp_path)
            
            # Update filename for all features
            ext = Path(file.filename).suffix
            # We use the provided name as the identifier
            # In the CSV, 'filename' column is used as the label
            filename = f"{name}{ext}" 
            
            for feat in features_list:
                feat['filename'] = filename
            
            # Append to CSV
            df = pd.DataFrame(features_list)
            
            # Ensure columns are in correct order (filename first)
            cols = ['filename'] + [c for c in df.columns if c != 'filename']
            df = df[cols]
            
            if not CSV_PATH.exists():
                df.to_csv(CSV_PATH, index=False)
            else:
                df.to_csv(CSV_PATH, mode='a', header=False, index=False)
                
            return jsonify({'status': 'success', 'message': f'Features extracted for {name} ({len(features_list)} samples)'})
            
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
        
        # Extract features manually to avoid augmentation and ensure consistency
        try:
            image = cv2.imread(str(temp_path), cv2.IMREAD_UNCHANGED)
            if image is None:
                raise ValueError("Cannot read image")
                
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            gray = gray.astype(np.uint8)
            
            # Resize and Equalize
            resized = cv2.resize(gray, extractor.win_size)
            processed = cv2.equalizeHist(resized)
            
            # Compute HOG
            hog_feats = extractor.hog.compute(processed)
            
            if hog_feats is None:
                raise ValueError("Could not extract HOG features")
                
            probe_features = hog_feats.flatten()
            
        except Exception as e:
            return jsonify({'status': 'error', 'message': f"Extraction error: {str(e)}"})
        
        results = {}
        
        # 1. Math Matcher Identification
        # MathMatcher might not be optimized for HOG features but we'll try
        if CSV_PATH.exists():
            try:
                labels, db_features = load_database(CSV_PATH)
                if len(labels) > 0:
                    math_matcher = MathMatcher()
                    math_matcher.fit(db_features, labels)
                    # HOG features are large, distance thresholds might need adjustment
                    # For now we use the default or what was there
                    math_id, math_conf, _ = math_matcher.identify(probe_features, min_absolute=70.0)
                    
                    results['math'] = {
                        'identified': math_id if math_conf['absolute'] >= 70.0 else "Unknown",
                        'confidence': math_conf['absolute'],
                        'gap': math_conf['gap']
                    }
                else:
                     results['math'] = {'error': 'Database empty'}
            except Exception as e:
                results['math'] = {'error': f"Math Matcher error: {str(e)}"}
        else:
            results['math'] = {'error': 'Database not found'}
        
        # 2. ML Matcher Identification
        if ML_MODEL_PATH.exists():
            try:
                ml_matcher = MLMatcher()
                ml_matcher.load(ML_MODEL_PATH)
                ml_id, ml_conf, _ = ml_matcher.identify(probe_features, min_confidence=70.0)
                
                results['ml'] = {
                    'identified': ml_id if ml_conf['confidence'] >= 70.0 else "Unknown",
                    'confidence': ml_conf['confidence'],
                    'probability': ml_conf['probability']
                }
            except Exception as e:
                results['ml'] = {'error': f"ML Matcher error: {str(e)}"}
        else:
            results['ml'] = {'error': 'Model not trained'}
            
        return jsonify({'status': 'success', 'results': results})
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=1235, debug=True)
