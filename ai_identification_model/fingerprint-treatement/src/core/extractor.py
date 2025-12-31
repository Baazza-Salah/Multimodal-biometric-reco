import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional

# --- Configuration ---
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / 'data' / 'archive' / 'SOCOFing' / 'Real'
OUTPUT_DIR = BASE_DIR / 'data'
MODEL_DIR = BASE_DIR / 'models'

# Ensure directories exist
for folder in [DATA_DIR, OUTPUT_DIR, MODEL_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

def get_files_from_directory(directory: Path, extensions: List[str]) -> List[Path]:
    """Retrieves all files with specified extensions from a directory."""
    files = []
    if not directory.exists():
        return []
    for ext in extensions:
        files.extend(directory.glob(f'*{ext}'))
        files.extend(directory.glob(f'*{ext.upper()}'))
    return sorted(files)

def safe_process(func, *args, **kwargs):
    """Safely executes a function and handles exceptions."""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        print(f"Error processing: {e}")
        return None

class FingerprintFeatureExtractor:
    """
    Extracteur de caractéristiques d'empreintes digitales.
    """

    def __init__(self):
        # Initialize HOG descriptor
        # WinSize: (64, 128) - standard for HOG
        # BlockSize: (16, 16)
        # BlockStride: (8, 8)
        # CellSize: (8, 8)
        # NBins: 9
        self.win_size = (64, 128)
        self.hog = cv2.HOGDescriptor(
            self.win_size,
            (16, 16),
            (8, 8),
            (8, 8),
            9
        )

    def preprocess_fingerprint(self, image: np.ndarray) -> np.ndarray:
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Ensure uint8 type
        gray = gray.astype(np.uint8)

        # Resize to fixed size for HOG
        resized = cv2.resize(gray, self.win_size)

        # Histogram equalization
        equalized = cv2.equalizeHist(resized)
        
        return equalized

    def augment_image(self, image: np.ndarray) -> List[np.ndarray]:
        """Generate augmented versions of the image."""
        augmented = [image]
        
        rows, cols = image.shape
        
        # 1. Rotations
        for angle in [-15, -10, -5, 5, 10, 15]:
            M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
            dst = cv2.warpAffine(image, M, (cols, rows), borderMode=cv2.BORDER_REPLICATE)
            augmented.append(dst)
            
        # 2. Noise injection (for original and some rotations)
        noisy_images = []
        for img in augmented[:3]: # Apply to original and first 2 rotations
            noise = np.random.normal(0, 10, img.shape).astype(np.uint8)
            noisy_img = cv2.add(img, noise)
            noisy_images.append(noisy_img)
            
        augmented.extend(noisy_images)
        
        return augmented

    def extract_features(self, image_path: Path) -> List[dict]:
        # Read image
        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")

        # Preprocess original to get base for augmentation
        # Note: preprocess_fingerprint does resizing, so we should augment AFTER resize 
        # or BEFORE. Let's do it BEFORE to simulate real variations, but we need grayscale first.
        
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        gray = gray.astype(np.uint8)
        
        # Generate augmented images
        images_to_process = self.augment_image(gray)
        
        all_features = []
        
        for idx, img in enumerate(images_to_process):
            # Resize and Equalize (part of preprocess)
            resized = cv2.resize(img, self.win_size)
            processed = cv2.equalizeHist(resized)
            
            # Compute HOG
            hog_feats = self.hog.compute(processed)
            
            if hog_feats is None:
                continue
                
            hog_feats = hog_feats.flatten()
            
            # Create feature dict
            # For augmented images, we keep the same filename so they share the label
            # The label extraction logic relies on the filename
            features = {'filename': image_path.name}
            
            for i, val in enumerate(hog_feats):
                features[f'f{i}'] = val
                
            all_features.append(features)

        return all_features


from joblib import Parallel, delayed
import multiprocessing

def process_fingerprints(input_dir: Path, output_path: Path):
    """
    Traite toutes les empreintes du dossier et exporte les features en CSV.
    """
    print("\n" + "="*60)
    print("TRAITEMENT DES EMPREINTES DIGITALES (AVEC AUGMENTATION)")
    print("="*60)

    extractor = FingerprintFeatureExtractor()
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif']
    image_files = get_files_from_directory(input_dir, image_extensions)

    print(f"Nombre d'images trouvées: {len(image_files)}")
    
    num_cores = multiprocessing.cpu_count()
    print(f"Utilisation de {num_cores} coeurs pour le traitement parallèle...")

    # Define a helper function for parallel execution
    def process_single_image(img_path):
        # Initialize extractor inside the worker to avoid pickling issues
        local_extractor = FingerprintFeatureExtractor()
        return safe_process(local_extractor.extract_features, img_path)

    # Run in parallel
    results = Parallel()(
        delayed(process_single_image)(img_path) 
        for img_path in image_files
    )
    
    # Flatten results
    all_features = []
    for res in results:
        if res:
            all_features.extend(res)

    print(f"\nTraitement terminé. {len(all_features)} échantillons générés.")

    if all_features:
        df = pd.DataFrame(all_features)
        df.to_csv(output_path, index=False)
        print(f"\n✓ Export réussi: {output_path}")
        print(f"  Shape: {df.shape}")
        print(f"  Colonnes: {list(df.columns)}")
    else:
        print("\n⚠ Aucune caractéristique extraite")


def main():
    """Main extraction pipeline."""
    input_dir = DATA_DIR
    output_path = OUTPUT_DIR / 'features.csv'

    print(f"Input Directory: {input_dir}")
    print(f"Output File: {output_path}")

    try:
        process_fingerprints(input_dir, output_path)
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
