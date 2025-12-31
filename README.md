# Biometric Face Identification System

This project is an advanced AI-based biometric identification system that uses facial feature extraction and machine learning to identify individuals. It supports both geometric feature matching and ML-based classification (SVM/KNN).

## 📂 Project Structure

### `src/`
Contains the source code for the project.
- **`core/`**: Core logic for feature extraction and matching.
    - **`extractor.py`**: The main extraction engine (MediaPipe/OpenCV).
    - **`matchers/`**: Contains `math_matcher.py` and `ml_matcher.py`.
- **`web/`**: The Flask web application.
    - **`app.py`**: The backend server.
    - **`templates/`**: HTML frontend files.

### `models/`
Stores the trained AI models.
- **`face_landmarker.task`**: MediaPipe model for landmark detection.
- **`trained_model.pkl`**: The saved trained ML model.

### `scripts/`
CLI scripts for management and testing.
- **`train.py`**: Script to train the ML model.
- **`identify_cli.py`**: CLI tool to test ML identification.
- **`identify_math.py`**: CLI tool to test Math identification.

### `data/`
Stores the database and images.
- **`features.csv`**: The main database file containing extracted features.
- **`images/`**: Directory for raw images.
- **`test/`**: Directory for test images.

---

## 📦 Installation

1.  **Clone the repository** (if you haven't already).
2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

---

## 🚀 How to Use

### 1. Web Application (Recommended)
The web app provides a complete interface for registration and identification.

**Run the App:**
```bash
python src/web/app.py
```
Open your browser at: **http://localhost:1234**

**Features:**
- **Register Tab**: 
  1. Enter a name (e.g., "John").
  2. Click "Open Camera".
  3. Click "Start Registration". The system will capture 20 face samples (1 per second), extract features, save them to the CSV, and automatically retrain the ML model.
- **Identify Tab**:
  - **Upload Image**: Select an image file to identify.
  - **Use Camera**: Capture a photo in real-time to identify.
  - **Results**: Displays identification results from both the Math Matcher and ML Matcher side-by-side.

### 2. Terminal / CLI Usage

**A. Extract Features from a Folder of Images:**
If you have a folder of images (e.g., `data/images/`) that you want to add to the database:
```bash
python src/core/extractor.py
```
*This will process all images in `data/images/` and update `data/features.csv`.*

**B. Train the ML Model:**
After adding new features to the CSV (manually or via extractor), you must retrain the model:
```bash
python scripts/train.py
```

**C. Identify a Person (ML Matcher):**
To test identification using the trained Machine Learning model:
```bash
python scripts/identify_cli.py data/test/messi.png
```

**D. Identify a Person (Math Matcher):**
To test identification using the geometric distance algorithm:
```bash
python scripts/identify_math.py data/test/messi.png
```

---

## 🛠️ Technologies Used

### 1. Feature Extraction (`src/core/`)
*   **MediaPipe (Google)**: Used for high-precision **Face Landmark Detection**.
    *   *Why?* It provides 478 3D face landmarks in real-time with superior accuracy compared to older methods like Haar Cascades or Dlib.
*   **OpenCV (`cv2`)**: Used for image processing, face alignment, and normalization.
    *   *Why?* The industry standard for computer vision tasks, allowing efficient image manipulation (rotation, cropping, resizing).
*   **Pandas**: Used for structuring and saving feature data to CSV.
    *   *Why?* Makes handling tabular data (features per person) easy and allows for quick database operations.

### 2. Identification Engine (`src/core/matchers/`)
*   **Scikit-learn (`sklearn`)**: The core Machine Learning library.
    *   *Why?* Provides robust implementations of **SVM (Support Vector Machines)**, **KNN (K-Nearest Neighbors)**, and **Random Forest**. It also handles data preprocessing (StandardScaler) and model evaluation (GridSearch).
*   **NumPy**: Used for vector mathematics and geometric calculations.
    *   *Why?* Essential for calculating Euclidean distances, angles, and ratios efficiently.

### 3. Web Application (`src/web/`)
*   **Flask**: A lightweight Python web framework.
    *   *Why?* Perfect for wrapping Python-based AI models into a RESTful API. It's simple, fast, and integrates seamlessly with our ML backend.
*   **HTML5 / JavaScript**: Used for the frontend interface.
    *   *Why?* Native browser APIs (`navigator.mediaDevices`) allow direct access to the webcam without external plugins, enabling real-time capture.

---

## 🏗️ Project Architecture

The system follows a modular pipeline architecture:

1.  **Input Layer**:
    *   **Webcam Feed** (Real-time) OR **Image Upload** (Static).
    *   Captured frames are sent to the backend.

2.  **Preprocessing & Extraction Layer** (`extractor.py`):
    *   **Detection**: MediaPipe detects the face mesh.
    *   **Normalization**: OpenCV aligns the face (eyes horizontal), crops to bounding box, and resizes to 224x224.
    *   **Feature Engineering**: 27 geometric features are calculated (e.g., `eye_width`, `nose_length`, `symmetry_score`).

3.  **Database Layer**:
    *   **`features.csv`**: Stores the normalized feature vectors for all enrolled users.

4.  **Identification Layer**:
    *   **Math Matcher**: Calculates weighted Euclidean distance between the probe vector and database vectors.
    *   **ML Matcher**: Uses a trained classifier (SVM/KNN) to predict the identity based on the feature vector.

5.  **Decision Layer**:
    *   **Thresholding**: Checks if the confidence score is **> 80%**.
    *   **Output**: Returns the Name if confident, otherwise "Unknown".

---

- **Confidence Threshold**: The system is configured to require an **80% confidence score** to positively identify a person. If the score is lower, it will return "Unknown".
- **Face Normalization**: All faces are automatically aligned and resized to 224x224 pixels before feature extraction to ensure consistency.
