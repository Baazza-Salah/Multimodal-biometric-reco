import pandas as pd
import numpy as np
from pathlib import Path
import re

CSV_PATH = Path('/home/luuketheone/Desktop/BIOID/ai_identification_model/fingerprint-treatement/data/features.csv')

def analyze_dataset():
    if not CSV_PATH.exists():
        print(f"File not found: {CSV_PATH}")
        return

    print("Loading CSV...")
    df = pd.read_csv(CSV_PATH)
    
    print(f"Total samples: {len(df)}")
    print(f"Number of features: {len(df.columns) - 1}")
    
    # Extract labels
    def get_label(filename):
        # Handle fingerprint format: 1__M_Left... -> 1
        if '__' in filename:
            return filename.split('__')[0]
        # Handle face format: messi1 -> messi
        base_name = re.sub(r'\d+$', '', filename)
        return base_name if base_name else filename

    df['label'] = df['filename'].apply(get_label)
    
    num_classes = df['label'].nunique()
    print(f"Number of classes (persons): {num_classes}")
    
    class_counts = df['label'].value_counts()
    print(f"Min samples per class: {class_counts.min()}")
    print(f"Max samples per class: {class_counts.max()}")
    print(f"Mean samples per class: {class_counts.mean():.2f}")
    
    print("\nTop 5 classes:")
    print(class_counts.head())

if __name__ == "__main__":
    analyze_dataset()
