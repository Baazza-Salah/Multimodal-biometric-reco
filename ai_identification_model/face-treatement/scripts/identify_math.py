# Enhanced Identification Script - Multi-Template Support
# Handles multiple enrollment images per person

import sys
from pathlib import Path

# Add parent directories to path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.append(str(PROJECT_ROOT))

from src.core.extractor import FaceFeatureExtractor, MODEL_PATH
from src.core.matchers.math_matcher import MathMatcher, load_database
import numpy as np

# Configuration
DATA_DIR = PROJECT_ROOT / 'data'
CSV_PATH = DATA_DIR / 'features.csv'

def extract_probe_features(image_path: Path) -> np.ndarray:
    """Extract features from probe image."""
    extractor = FaceFeatureExtractor(MODEL_PATH)
    features_dict = extractor.extract_features(image_path)
    extractor.close()
    
    # Remove filename and convert to array
    del features_dict['filename']
    return np.array(list(features_dict.values()), dtype='float32')

def identify(image_path: str):
    """Main identification function with multi-template support."""
    image_path = Path(image_path)
    
    if not image_path.exists():
        print(f"Error: Image not found: {image_path}")
        return
    
    # Load database
    labels, db_features = load_database(CSV_PATH)
    
    # Initialize and fit matcher
    matcher = MathMatcher()
    matcher.fit(db_features, labels)
    
    # Extract probe features
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
    print("IDENTIFICATION RESULTS")
    print("="*70)
    
    if identified != "Unknown":
        print(f"\n✓ IDENTIFIED: {identified.upper()}")
        print(f"\n  Confidence Metrics:")
        print(f"    Absolute Match: {confidence['absolute']:.1f}%")
        print(f"    Relative Score: {confidence['relative']:.1f}%")
        print(f"    Gap to 2nd: {confidence['gap']:.1f}%")
        print(f"    Best Distance: {confidence['distance']:.4f}")
        print(f"    Templates Used: {confidence['n_templates']}")
    else:
        print(f"\n✗ NOT IDENTIFIED")
        print(f"\n  Analysis:")
        print(f"    Best Match: {all_results[0]['person']}")
        print(f"    Absolute Score: {confidence['absolute']:.1f}%")
        print(f"    Relative Score: {confidence['relative']:.1f}%")
        print(f"    Gap to 2nd: {confidence['gap']:.1f}%")
        print(f"    Distance: {confidence['distance']:.4f}")
        if 'reason' in confidence:
            print(f"    Rejection Reason: {confidence['reason']}")
    
    # Show all candidates
    print("\n  All Candidates (Person-Level Aggregated Scores):")
    print("  " + "-"*66)
    for i, r in enumerate(all_results, 1):
        indicator = "  <<<" if r['person'] == identified else ""
        print(f"  {i:2}. {r['person']:15} | Abs: {r['absolute_score']:5.1f}% | "
              f"Rel: {r['relative_score']:5.1f}% | Dist: {r['best_distance']:.4f} | "
              f"T:{r['n_templates']}{indicator}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python identify.py <image_path>")
        print("Example: python identify.py ../data/test/probe.png")
    else:
        identify(sys.argv[1])
