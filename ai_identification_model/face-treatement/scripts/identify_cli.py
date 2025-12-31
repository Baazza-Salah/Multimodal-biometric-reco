# ML Model Identification Script
# Uses trained ML model for face identification

import sys
from pathlib import Path

# Add parent directories to path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.append(str(PROJECT_ROOT))

from src.core.extractor import FaceFeatureExtractor, MODEL_PATH as FACE_MODEL_PATH
from src.core.matchers.ml_matcher import MLMatcher
import numpy as np

# Configuration
MODEL_PATH = PROJECT_ROOT / 'models' / 'trained_model.pkl'

def extract_probe_features(image_path: Path) -> np.ndarray:
    """Extract features from probe image."""
    extractor = FaceFeatureExtractor(FACE_MODEL_PATH)
    features_dict = extractor.extract_features(image_path)
    extractor.close()
    
    del features_dict['filename']
    return np.array(list(features_dict.values()), dtype='float32')

def identify(image_path: str):
    """Identify using trained ML model."""
    image_path = Path(image_path)
    
    if not image_path.exists():
        print(f"Error: Image not found: {image_path}")
        return
    
    if not MODEL_PATH.exists():
        print(f"Error: Trained model not found: {MODEL_PATH}")
        print("Please run: python train_model.py first")
        return
    
    # Load model
    print("Loading trained ML model...")
    matcher = MLMatcher()
    matcher.load(MODEL_PATH)
    
    # Extract features
    print(f"\nAnalyzing: {image_path.name}")
    try:
        probe_features = extract_probe_features(image_path)
    except Exception as e:
        print(f"Error: {e}")
        return
    
    # Identify
    identified, confidence, all_results = matcher.identify(probe_features)
    
    # Display results
    print("\n" + "="*70)
    print("ML IDENTIFICATION RESULTS")
    print("="*70)
    
    print(f"\nModel: {confidence['algorithm'].upper()}")
    print(f"Training Accuracy: {confidence['cv_score']:.1f}%")
    
    if identified != "Unknown":
        print(f"\n✓ IDENTIFIED: {identified.upper()}")
        print(f"\n  Confidence Metrics:")
        print(f"    Confidence: {confidence['confidence']:.1f}%")
        print(f"    Probability: {confidence['probability']:.4f}")
        print(f"    Gap to 2nd: {confidence['gap']:.1f}%")
    else:
        print(f"\n✗ NOT IDENTIFIED")
        print(f"\n  Analysis:")
        print(f"    Best Match: {all_results[0]['person']}")
        print(f"    Confidence: {confidence['confidence']:.1f}%")
        print(f"    Gap to 2nd: {confidence['gap']:.1f}%")
        if 'reason' in confidence:
            print(f"    Rejection: {confidence['reason']}")
    
    # Show all predictions
    print("\n  All Predictions (sorted by probability):")
    print("  " + "-"*66)
    for i, r in enumerate(all_results, 1):
        indicator = "  <<<" if r['person'] == identified else ""
        print(f"  {i:2}. {r['person']:15} | Confidence: {r['confidence']:5.1f}% | "
              f"Prob: {r['probability']:.4f}{indicator}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python identify_mlmodel.py <image_path>")
        print("Example: python identify_mlmodel.py ../data/test/probe.png")
        print("\nNote: Run 'python train_model.py' first to train the model")
    else:
        identify(sys.argv[1])
