# Biometric Identification Web App
# Flask backend for face registration and identification

import os
import sys
import cv2
import numpy as np
import base64
import pandas as pd
import uuid
from pathlib import Path
from flask import Flask, render_template, request, jsonify
from cryptography.fernet import Fernet

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
USERS_CSV_PATH = DATA_DIR / 'users.csv'
KEY_PATH = DATA_DIR / 'secret.key'
ML_MODEL_PATH = PROJECT_ROOT / 'models' / 'trained_model.pkl'

class EncryptionManager:
    def __init__(self, key_path):
        self.key_path = key_path
        self.key = self.load_or_generate_key()
        self.cipher_suite = Fernet(self.key)

    def load_or_generate_key(self):
        if self.key_path.exists():
            with open(self.key_path, 'rb') as key_file:
                return key_file.read()
        else:
            key = Fernet.generate_key()
            with open(self.key_path, 'wb') as key_file:
                key_file.write(key)
            return key

    def encrypt(self, data):
        return self.cipher_suite.encrypt(data.encode()).decode()

    def decrypt(self, token):
        return self.cipher_suite.decrypt(token.encode()).decode()

# Initialize encryption manager
encryption_manager = EncryptionManager(KEY_PATH)

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
        country = data.get('country', 'Unknown')
        user_uuid = data.get('uuid')
        count = data['count']
        
        if not user_uuid:
            return jsonify({'status': 'error', 'message': 'UUID is missing'})
        
        # Convert image
        image = base64_to_image(image_data)
        
        # Save temporary image for extractor
        temp_path = SCRIPT_DIR / 'temp_capture.png'
        cv2.imwrite(str(temp_path), image)
        
        # Extract features
        try:
            features = extractor.extract_features(temp_path)
            
            # Add filename with UUID and count (e.g., uuid_1.png)
            # We use the UUID as the base identifier
            filename = f"{user_uuid}_{count}.png"
            features['filename'] = filename
            
            # Append to Features CSV
            df = pd.DataFrame([features])
            
            if not CSV_PATH.exists():
                df.to_csv(CSV_PATH, index=False)
            else:
                df.to_csv(CSV_PATH, mode='a', header=False, index=False)
            
            # Save Encrypted User Data (only need to do this once, but doing it every time ensures it's saved)
            # Or we can check if UUID exists in users.csv. 
            # For simplicity and robustness, we can read, check, and append if new.
            
            user_data = f"{name}|{country}"
            encrypted_data = encryption_manager.encrypt(user_data)
            
            user_entry = {'uuid': user_uuid, 'encrypted_data': encrypted_data}
            users_df = pd.DataFrame([user_entry])
            
            if not USERS_CSV_PATH.exists():
                users_df.to_csv(USERS_CSV_PATH, index=False)
            else:
                # Check if UUID already exists to avoid duplicates
                existing_users = pd.read_csv(USERS_CSV_PATH)
                if user_uuid not in existing_users['uuid'].values:
                    users_df.to_csv(USERS_CSV_PATH, mode='a', header=False, index=False)
                
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
        
        # Decrypt Name for Math Matcher
        math_name = "Unknown"
        if math_id != "Unknown":
            try:
                users_df = pd.read_csv(USERS_CSV_PATH)
                user_row = users_df[users_df['uuid'] == math_id]
                if not user_row.empty:
                    encrypted_data = user_row.iloc[0]['encrypted_data']
                    decrypted_data = encryption_manager.decrypt(encrypted_data)
                    math_name = decrypted_data.split('|')[0]  # Name|Country
                else:
                    math_name = f"UUID: {math_id} (Name not found)"
            except Exception as e:
                math_name = f"Error decrypting: {str(e)}"

        results['math'] = {
            'identified': math_name if math_conf['absolute'] >= 70.0 else "Unknown",
            'confidence': math_conf['absolute'],
            'gap': math_conf['gap']
        }
        
        # 2. ML Matcher Identification
        if ML_MODEL_PATH.exists():
            ml_matcher = MLMatcher()
            ml_matcher.load(ML_MODEL_PATH)
            ml_id, ml_conf, _ = ml_matcher.identify(probe_features, min_confidence=70.0)
            
            # Decrypt Name for ML Matcher
            ml_name = "Unknown"
            if ml_id != "Unknown":
                try:
                    users_df = pd.read_csv(USERS_CSV_PATH)
                    user_row = users_df[users_df['uuid'] == ml_id]
                    if not user_row.empty:
                        encrypted_data = user_row.iloc[0]['encrypted_data']
                        decrypted_data = encryption_manager.decrypt(encrypted_data)
                        ml_name = decrypted_data.split('|')[0]
                    else:
                        ml_name = f"UUID: {ml_id} (Name not found)"
                except Exception as e:
                    ml_name = f"Error decrypting: {str(e)}"
            
            results['ml'] = {
                'identified': ml_name if ml_conf['confidence'] >= 70.0 else "Unknown",
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
