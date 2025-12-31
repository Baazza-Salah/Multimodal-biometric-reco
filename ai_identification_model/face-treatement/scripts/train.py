# ML Model Training Script
# Trains the ML model on the feature database

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.append(str(PROJECT_ROOT))

from src.core.matchers.ml_matcher import MLMatcher, load_database

# Configuration
DATA_DIR = PROJECT_ROOT / 'data'
CSV_PATH = DATA_DIR / 'features.csv'
MODEL_PATH = PROJECT_ROOT / 'models' / 'trained_model.pkl'

def train_model():
    """Train and save the ML model."""
    print("="*70)
    print("ML MODEL TRAINING")
    print("="*70)
    
    # Load database
    print(f"\nLoading database from: {CSV_PATH}")
    labels, features = load_database(CSV_PATH)
    print(f"Loaded {len(labels)} templates")
    
    # Initialize matcher
    matcher = MLMatcher(algorithm='auto')  # Auto-selects best algorithm
    
    # Train
    results = matcher.fit(features, labels)
    
    print("\n" + "-"*30)
    print("TRAINING RESULTS")
    print("-"*30)
    
    # Print all scores
    print("\nModel Performance:")
    for algo, score in results['all_scores'].items():
        print(f"  - {algo.upper():<15}: {score:.2%}")
    
    print(f"\n✓ SELECTED MODEL: {results['best_algo'].upper()}")
    print(f"  Accuracy: {results['best_score']:.2%}")
    print(f"  Best Params: {results['best_params']}")
    
    # Save model
    matcher.save(MODEL_PATH)
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"Model saved to: {MODEL_PATH}")
    print(f"\nYou can now use: python identify_mlmodel.py <image_path>")

if __name__ == "__main__":
    train_model()
