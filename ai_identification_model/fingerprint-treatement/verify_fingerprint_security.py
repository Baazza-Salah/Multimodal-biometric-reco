import sys
from pathlib import Path
import pandas as pd
import io
from unittest.mock import MagicMock

# Add parent directories to path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR
sys.path.append(str(PROJECT_ROOT))

from src.web.app import app, EncryptionManager, KEY_PATH, USERS_CSV_PATH, CSV_PATH, extractor

def verify_security():
    print("Starting Security Verification...")
    
    # 1. Test EncryptionManager
    print("\nTesting EncryptionManager...")
    manager = EncryptionManager(KEY_PATH)
    original_text = "Test Name|Test Country"
    encrypted = manager.encrypt(original_text)
    decrypted = manager.decrypt(encrypted)
    
    assert original_text == decrypted
    print("Encryption/Decryption successful!")
    
    # 2. Test Registration Flow (Mocked)
    print("\nTesting Registration Flow...")
    
    # Mock extractor to avoid image processing errors
    extractor.extract_features = MagicMock(return_value=[{'feature1': 0.1, 'feature2': 0.2}])
    
    client = app.test_client()
    
    uuid = "test-fingerprint-uuid"
    payload = {
        'name': 'Test User',
        'country': 'Testland',
        'uuid': uuid
    }
    
    # Create a dummy file
    data = {
        'name': 'Test User',
        'country': 'Testland',
        'uuid': uuid,
        'file': (io.BytesIO(b"dummy image data"), 'test.bmp')
    }
    
    response = client.post('/extract_feature', data=data, content_type='multipart/form-data')
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
    
    # 3. Test Identification Logic (Mocked)
    print("\nTesting Identification Logic (Mock)...")
    
    # We simulate the identification process by manually checking if we can retrieve the name
    # given the UUID (which would be returned by the matcher)
    
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
            raise Exception("User not found in CSV for identification test")
            
    except Exception as e:
        print(f"Identification logic test failed: {e}")
        raise

if __name__ == "__main__":
    verify_security()
