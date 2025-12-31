# Enhanced Face Feature Extraction with Normalization
# Detects, aligns, normalizes faces before extracting features

import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple
import warnings
import math

warnings.filterwarnings('ignore')

# MediaPipe Tasks API imports
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision

# --- Configuration ---
# --- Configuration ---
BASE_DIR = Path(__file__).resolve().parent.parent.parent

DATA_DIR = BASE_DIR / 'data' / 'images'
OUTPUT_DIR = BASE_DIR / 'data'
MODEL_PATH = BASE_DIR / 'models' / 'face_landmarker.task'

# Normalization parameters
NORMALIZED_FACE_SIZE = (224, 224)  # Standard size for all faces
EYE_Y_POSITION = 0.35  # Eyes at 35% from top

# Ensure directories exist
for folder in [DATA_DIR, OUTPUT_DIR]:
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


class FaceFeatureExtractor:
    """
    Enhanced Face Feature Extractor with normalization.
    
    Pipeline:
    1. Detect face landmarks
    2. Align face based on eye positions
    3. Normalize to standard size
    4. Extract geometric features
    """

    # Key landmark indices
    LANDMARKS = {
        'nose_tip': 1,
        'nose_bridge': 6,
        'nose_left': 129,
        'nose_right': 358,
        'left_eye_inner': 133,
        'left_eye_outer': 33,
        'left_eye_top': 159,
        'left_eye_bottom': 145,
        'right_eye_inner': 362,
        'right_eye_outer': 263,
        'right_eye_top': 386,
        'right_eye_bottom': 374,
        'left_eyebrow_inner': 107,
        'left_eyebrow_outer': 70,
        'right_eyebrow_inner': 336,
        'right_eyebrow_outer': 300,
        'mouth_left': 61,
        'mouth_right': 291,
        'mouth_top': 0,
        'mouth_bottom': 17,
        'upper_lip_top': 13,
        'lower_lip_bottom': 14,
        'chin': 152,
        'forehead': 10,
        'left_cheek': 234,
        'right_cheek': 454,
        'left_jaw': 172,
        'right_jaw': 397,
    }

    def __init__(self, model_path: Path):
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        base_options = mp_tasks.BaseOptions(model_asset_path=str(model_path))
        options = mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1
        )
        self.detector = mp_vision.FaceLandmarker.create_from_options(options)

    def _normalize_face(self, image: np.ndarray, landmarks) -> Tuple[np.ndarray, object]:
        """
        Normalize face: align based on eyes and resize to standard size.
        
        Returns:
            (normalized_image, updated_landmarks)
        """
        h, w = image.shape[:2]
        
        # Get eye positions
        left_eye = landmarks[self.LANDMARKS['left_eye_outer']]
        right_eye = landmarks[self.LANDMARKS['right_eye_outer']]
        
        left_eye_px = np.array([left_eye.x * w, left_eye.y * h])
        right_eye_px = np.array([right_eye.x * w, right_eye.y * h])
        
        # Calculate rotation angle to align eyes horizontally
        delta = right_eye_px - left_eye_px
        angle = math.degrees(math.atan2(delta[1], delta[0]))
        
        # Calculate eye center (needs to be float tuple for cv2)
        eye_center = (left_eye_px + right_eye_px) / 2
        eye_center_tuple = (float(eye_center[0]), float(eye_center[1]))
        
        # Rotation matrix
        rot_matrix = cv2.getRotationMatrix2D(eye_center_tuple, angle, 1.0)
        
        # Rotate image
        rotated = cv2.warpAffine(image, rot_matrix, (w, h), flags=cv2.INTER_CUBIC)
        
        # Calculate new eye positions after rotation
        left_eye_rotated = rot_matrix @ np.array([left_eye_px[0], left_eye_px[1], 1])
        right_eye_rotated = rot_matrix @ np.array([right_eye_px[0], right_eye_px[1], 1])
        
        # Calculate face bounding box based on landmarks
        all_x = [lm.x * w for lm in landmarks]
        all_y = [lm.y * h for lm in landmarks]
        
        # Rotate all points
        rotated_points = []
        for x, y in zip(all_x, all_y):
            pt = rot_matrix @ np.array([x, y, 1])
            rotated_points.append(pt)
        
        rotated_x = [pt[0] for pt in rotated_points]
        rotated_y = [pt[1] for pt in rotated_points]
        
        # Calculate bounding box with padding
        min_x = max(0, int(min(rotated_x)) - 20)
        max_x = min(w, int(max(rotated_x)) + 20)
        min_y = max(0, int(min(rotated_y)) - 40)
        max_y = min(h, int(max(rotated_y)) + 20)
        
        # Crop face
        face_crop = rotated[min_y:max_y, min_x:max_x]
        
        # Resize to standard size
        normalized = cv2.resize(face_crop, NORMALIZED_FACE_SIZE, interpolation=cv2.INTER_CUBIC)
        
        # Re-detect landmarks on normalized face
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=normalized)
        result = self.detector.detect(mp_image)
        
        if not result.face_landmarks:
            # If detection fails on normalized image, return original landmarks
            return normalized, landmarks
        
        return normalized, result.face_landmarks[0]

    def _get_point(self, landmarks, name: str) -> np.ndarray:
        """Get 3D coordinates of a named landmark."""
        idx = self.LANDMARKS[name]
        lm = landmarks[idx]
        return np.array([lm.x, lm.y, lm.z])

    def _distance(self, p1: np.ndarray, p2: np.ndarray) -> float:
        """Calculate Euclidean distance between two points."""
        return np.linalg.norm(p1 - p2)

    def _angle(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        """Calculate angle at p2 formed by p1-p2-p3 in degrees."""
        v1 = p1 - p2
        v2 = p3 - p2
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        return math.degrees(math.acos(cos_angle))

    def extract_features(self, image_path: Path) -> Dict:
        """Extracts comprehensive facial features from an image with normalization."""
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")
        
        # Initial detection
        mp_image = mp.Image.create_from_file(str(image_path))
        result = self.detector.detect(mp_image)
        
        if not result.face_landmarks:
            raise ValueError("No face detected in image")

        initial_landmarks = result.face_landmarks[0]
        
        # Normalize face
        normalized_face, landmarks = self._normalize_face(image, initial_landmarks)
        
        features = {'filename': image_path.name}

        # === REFERENCE MEASUREMENTS ===
        left_eye_outer = self._get_point(landmarks, 'left_eye_outer')
        right_eye_outer = self._get_point(landmarks, 'right_eye_outer')
        chin = self._get_point(landmarks, 'chin')
        forehead = self._get_point(landmarks, 'forehead')
        
        face_width = self._distance(left_eye_outer, right_eye_outer)
        face_height = self._distance(forehead, chin)
        
        if face_width < 0.01 or face_height < 0.01:
            raise ValueError("Face too small")

        # === NORMALIZED DISTANCES ===
        left_eye_inner = self._get_point(landmarks, 'left_eye_inner')
        right_eye_inner = self._get_point(landmarks, 'right_eye_inner')
        
        features['eye_width_left'] = self._distance(left_eye_inner, left_eye_outer) / face_width
        features['eye_width_right'] = self._distance(right_eye_inner, right_eye_outer) / face_width
        features['eye_spacing'] = self._distance(left_eye_inner, right_eye_inner) / face_width
        
        left_eye_top = self._get_point(landmarks, 'left_eye_top')
        left_eye_bottom = self._get_point(landmarks, 'left_eye_bottom')
        right_eye_top = self._get_point(landmarks, 'right_eye_top')
        right_eye_bottom = self._get_point(landmarks, 'right_eye_bottom')
        
        features['eye_height_left'] = self._distance(left_eye_top, left_eye_bottom) / face_height
        features['eye_height_right'] = self._distance(right_eye_top, right_eye_bottom) / face_height
        
        nose_tip = self._get_point(landmarks, 'nose_tip')
        nose_bridge = self._get_point(landmarks, 'nose_bridge')
        nose_left = self._get_point(landmarks, 'nose_left')
        nose_right = self._get_point(landmarks, 'nose_right')
        
        features['nose_length'] = self._distance(nose_bridge, nose_tip) / face_height
        features['nose_width'] = self._distance(nose_left, nose_right) / face_width
        
        mouth_left = self._get_point(landmarks, 'mouth_left')
        mouth_right = self._get_point(landmarks, 'mouth_right')
        mouth_top = self._get_point(landmarks, 'mouth_top')
        mouth_bottom = self._get_point(landmarks, 'mouth_bottom')
        
        features['mouth_width'] = self._distance(mouth_left, mouth_right) / face_width
        features['mouth_height'] = self._distance(mouth_top, mouth_bottom) / face_height
        
        left_cheek = self._get_point(landmarks, 'left_cheek')
        right_cheek = self._get_point(landmarks, 'right_cheek')
        left_jaw = self._get_point(landmarks, 'left_jaw')
        right_jaw = self._get_point(landmarks, 'right_jaw')
        
        features['cheek_width'] = self._distance(left_cheek, right_cheek) / face_width
        features['jaw_width'] = self._distance(left_jaw, right_jaw) / face_width
        
        left_eb_inner = self._get_point(landmarks, 'left_eyebrow_inner')
        left_eb_outer = self._get_point(landmarks, 'left_eyebrow_outer')
        right_eb_inner = self._get_point(landmarks, 'right_eyebrow_inner')
        right_eb_outer = self._get_point(landmarks, 'right_eyebrow_outer')
        
        features['eyebrow_length_left'] = self._distance(left_eb_inner, left_eb_outer) / face_width
        features['eyebrow_length_right'] = self._distance(right_eb_inner, right_eb_outer) / face_width
        
        # === VERTICAL PROPORTIONS ===
        features['forehead_to_eyebrow'] = self._distance(forehead, left_eb_inner) / face_height
        features['eyebrow_to_eye'] = self._distance(left_eb_inner, left_eye_top) / face_height
        features['eye_to_nose'] = self._distance(left_eye_bottom, nose_tip) / face_height
        features['nose_to_mouth'] = self._distance(nose_tip, mouth_top) / face_height
        features['mouth_to_chin'] = self._distance(mouth_bottom, chin) / face_height
        
        # === ANGULAR FEATURES ===
        eye_center_left = (left_eye_inner + left_eye_outer) / 2
        eye_center_right = (right_eye_inner + right_eye_outer) / 2
        eye_line = eye_center_right - eye_center_left
        features['eye_tilt'] = math.degrees(math.atan2(eye_line[1], eye_line[0]))
        
        face_center_x = (left_eye_outer[0] + right_eye_outer[0]) / 2
        features['nose_deviation'] = (nose_tip[0] - face_center_x) / face_width
        
        mouth_line = mouth_right - mouth_left
        features['mouth_tilt'] = math.degrees(math.atan2(mouth_line[1], mouth_line[0]))
        
        # === SYMMETRY ===
        features['eye_symmetry'] = abs(features['eye_width_left'] - features['eye_width_right'])
        features['eyebrow_symmetry'] = abs(features['eyebrow_length_left'] - features['eyebrow_length_right'])
        
        # === RATIOS ===
        features['face_aspect_ratio'] = face_width / face_height
        features['eye_mouth_ratio'] = features['eye_spacing'] / features['mouth_width']
        features['nose_mouth_ratio'] = features['nose_width'] / features['mouth_width']
        features['upper_lower_face'] = features['nose_to_mouth'] / (features['mouth_to_chin'] + 1e-8)

        return features

    def close(self):
        if hasattr(self, 'detector') and self.detector:
            self.detector.close()


def process_faces(input_dir: Path, output_path: Path, model_path: Path):
    """Processes all faces with normalization and exports features to CSV."""
    extractor = FaceFeatureExtractor(model_path)
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = get_files_from_directory(input_dir, image_extensions)

    print(f"Processing {len(image_files)} images with face normalization...")

    all_features = []

    for i, img_path in enumerate(image_files, 1):
        print(f"[{i}/{len(image_files)}] {img_path.name}... ", end='')

        try:
            features = extractor.extract_features(img_path)
            all_features.append(features)
            print("✓")
        except Exception as e:
            print(f"✗ ({e})")

    extractor.close()

    if all_features:
        df = pd.DataFrame(all_features)
        df.to_csv(output_path, index=False)
        print(f"\n✓ Saved {len(all_features)} normalized feature sets")
        print(f"  Features per face: {len(df.columns) - 1}")
    else:
        print("\n⚠ No features extracted")


def main():
    """Main extraction pipeline."""
    faces_input = DATA_DIR
    faces_output = OUTPUT_DIR / 'features.csv'

    try:
        process_faces(faces_input, faces_output, MODEL_PATH)
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()