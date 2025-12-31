import sys
import cv2
import numpy as np
from pathlib import Path

# Add project root to path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.append(str(PROJECT_ROOT))

from src.core.extractor import FingerprintFeatureExtractor
from src.core.matchers.ml_matcher import MLMatcher

def identify_fingerprint(image_path):
    """Identify a fingerprint from an image file."""
    print("="*60)
    print("FINGERPRINT IDENTIFICATION")
    print("="*60)
    
    image_path = Path(image_path)
    if not image_path.exists():
        print(f"Error: Image not found at {image_path}")
        return

    print(f"Processing image: {image_path.name}")
    
    # 1. Extract Features
    print("Extracting features...")
    extractor = FingerprintFeatureExtractor()
    
    # We need to extract features just like in the training phase
    # The extractor returns a list of features (including augmentations)
    # For identification, we just need the features of the original image
    # But the extractor logic is a bit coupled with augmentation
    
    # Let's manually use the extractor logic for a single image without augmentation loop if possible
    # Or just take the first result which corresponds to the original image
    
    try:
        # Read image
        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")
            
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        gray = gray.astype(np.uint8)
        
        # Resize and Equalize (same as in extractor.py)
        resized = cv2.resize(gray, extractor.win_size)
        processed = cv2.equalizeHist(resized)
        
        # Compute HOG
        hog_feats = extractor.hog.compute(processed)
        
        if hog_feats is None:
            print("Error: Could not extract HOG features")
            return
            
        features = hog_feats.flatten()
        
        print(f"Extracted {len(features)} features")
        
    except Exception as e:
        print(f"Error during extraction: {e}")
        return

    # 2. Load Model
    model_path = PROJECT_ROOT / 'models' / 'trained_model.pkl'
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        return
        
    print(f"Loading model from: {model_path}")
    matcher = MLMatcher()
    matcher.load(model_path)
    
    # 3. Identify
    print("Identifying...")
    identified_person, confidence_data, all_results = matcher.identify(features)
    
    print("\n" + "-"*30)
    print("IDENTIFICATION RESULT")
    print("-"*30)
    
    if identified_person == "Unknown":
        print(f"⚠ Result: UNKNOWN")
        print(f"  Reason: {confidence_data.get('reason', 'Low confidence')}")
    else:
        print(f"✓ Result: MATCH FOUND")
        print(f"  Identity: {identified_person}")
        print(f"  Confidence: {confidence_data['confidence']:.2f}%")
        print(f"  Probability: {confidence_data['probability']:.4f}")
    
    print("\nTop Matches:")
    for res in all_results[:3]:
        print(f"  - {res['person']}: {res['confidence']:.2f}%")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python identify.py <image_path>")
    else:
        identify_fingerprint(sys.argv[1])
