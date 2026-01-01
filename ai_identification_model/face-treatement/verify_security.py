
import base64
import cv2
import numpy as np
import os
import sys
import pandas as pd
from pathlib import Path

# Add parent directories to path for imports
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.web.app import EncryptionManager, KEY_PATH, USERS_CSV_PATH

def test_encryption():
    print("Testing EncryptionManager...")
    manager = EncryptionManager(KEY_PATH)
    original_text = "Test Name|Test Country"
    encrypted = manager.encrypt(original_text)
    decrypted = manager.decrypt(encrypted)
    
    assert original_text == decrypted
    print("Encryption/Decryption successful!")

def test_registration_flow():
    print("\nTesting Registration Flow...")
    url = "http://localhost:1234/extract_feature"
    
    # Create a dummy image
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    _, buffer = cv2.imencode('.jpg', img)
    img_str = base64.b64encode(buffer).decode('utf-8')
    
    uuid = "test-uuid-12345"
    payload = {
        "image": img_str,
        "name": "Test User",
        "country": "Testland",
        "uuid": uuid,
        "count": 1
    }
    
    try:
        from src.web.app import app, extractor
        from unittest.mock import MagicMock
        
        # Mock the extractor to avoid "No face detected" errors with dummy image
        extractor.extract_features = MagicMock(return_value={'feature1': 0.1, 'feature2': 0.2})
        
        client = app.test_client()
        
        response = client.post('/extract_feature', json=payload)
        assert response.status_code == 200
        
        json_data = response.get_json()
        if json_data['status'] != 'success':
            raise Exception(f"API Error: {json_data.get('message')}")
            
        print("Registration request successful!")
        
        # Verify users.csv
        if not USERS_CSV_PATH.exists():
             raise Exception(f"users.csv not found at {USERS_CSV_PATH}")
             
        df = pd.read_csv(USERS_CSV_PATH)
        user_row = df[df['uuid'] == uuid]
        assert not user_row.empty
        print("User saved to users.csv")
        
        # Verify encryption in csv
        encrypted_data = user_row.iloc[0]['encrypted_data']
        manager = EncryptionManager(KEY_PATH)
        decrypted = manager.decrypt(encrypted_data)
        assert decrypted == "Test User|Testland"
        print("Data in CSV is correctly encrypted and decryptable")
        
    except Exception as e:
        print(f"Registration test failed: {e}")

def test_identification_logic():
    print("\nTesting Identification Logic (Mock)...")
    # This is harder to test without a trained model and actual features.
    # But we can test the decryption part if we had a match.
    
    uuid = "test-uuid-12345"
    manager = EncryptionManager(KEY_PATH)
    
    # Simulate finding this UUID
    try:
        df = pd.read_csv(USERS_CSV_PATH)
        user_row = df[df['uuid'] == uuid]
        if not user_row.empty:
            encrypted_data = user_row.iloc[0]['encrypted_data']
            decrypted = manager.decrypt(encrypted_data)
            name = decrypted.split('|')[0]
            assert name == "Test User"
            print(f"Identification decryption successful! Found: {name}")
        else:
            print("User not found in CSV for identification test")
            
    except Exception as e:
        print(f"Identification logic test failed: {e}")

if __name__ == "__main__":
    test_encryption()
    test_registration_flow()
    test_identification_logic()
    
    # Clean up test data
    # os.remove(USERS_CSV_PATH) # Keep it for manual inspection if needed, or delete. 
    # For now I'll leave it but maybe I should clean up the test entry.
