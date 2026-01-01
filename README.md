# Multimodal Biometric Identification System

This project is an advanced AI-based multimodal biometric identification system that integrates **Face** and **Fingerprint** recognition. It leverages machine learning and geometric feature extraction to identify individuals securely and accurately.

## 🔒 Security & Privacy
- **UUID-Based Storage**: Biometric features are stored using unique UUIDs instead of personal names to ensure anonymity in the feature database.
- **Data Encryption**: Personal information (Name, Country) is **encrypted** using industry-standard cryptography before being stored. It is only decrypted momentarily during the identification process to display the result.

## 📂 Project Structure

The project is divided into two main modules within `ai_identification_model/`:

### 1. `face-treatement/`
Contains the Face Identification System.
- **`src/`**: Source code for extraction, matching, and the web app.
- **`data/`**: Stores encrypted user data (`users.csv`) and feature vectors (`features.csv`).
- **`models/`**: Trained ML models for face recognition.

### 2. `fingerprint-treatement/`
Contains the Fingerprint Identification System.
- **`src/`**: Source code for fingerprint processing and matching.
- **`data/`**: Stores encrypted user data and fingerprint features.
- **`models/`**: Trained ML models for fingerprint recognition.

---

## 📦 Installation

1.  **Clone the repository**:
    ```bash
    git clone <repository-url>
    cd Multimodal-biometic
    ```

2.  **Install Dependencies**:
    The project uses Python. You may need to install dependencies for each module.
    
    **For Face System:**
    ```bash
    cd ai_identification_model/face-treatement
    pip install -r requirements.txt
    ```

    **For Fingerprint System:**
    ```bash
    cd ../fingerprint-treatement
    pip install -r requirements.txt
    ```

---

## 🚀 How to Use

### 👤 Face Identification System

1.  **Navigate to the directory**:
    ```bash
    cd ai_identification_model/face-treatement
    ```

2.  **Run the Web App**:
    ```bash
    python src/web/app.py
    ```
    *The app will start on port `1234`.*

3.  **Open Browser**: Go to **http://localhost:1234**

4.  **Workflow**:
    - **Register**:
        - Enter **Name** and **Country**.
        - Click "Start Registration" to capture face samples via webcam.
        - **Save the displayed UUID**.
    - **Identify**:
        - Use the webcam or upload an image.
        - The system will match the face, retrieve the UUID, decrypt the data, and display the **Name**.

### 👆 Fingerprint Identification System

1.  **Navigate to the directory**:
    ```bash
    cd ai_identification_model/fingerprint-treatement
    ```

2.  **Run the Web App**:
    ```bash
    python src/web/app.py
    ```
    *The app will start on port `1235`.*

3.  **Open Browser**: Go to **http://localhost:1235**

4.  **Workflow**:
    - **Register**:
        - Enter **Name** and **Country**.
        - Upload a fingerprint image (BMP, PNG, JPG).
        - Click "Register Fingerprint".
        - **Save the displayed UUID**.
    - **Identify**:
        - Upload a probe fingerprint image.
        - Click "Identify Person".
        - The system will display the decrypted **Name**.

---

## 🛠️ Technologies Used

- **Core AI**:
    - **MediaPipe**: For high-precision 3D Face Landmark detection.
    - **OpenCV**: For image processing (face alignment, fingerprint enhancement).
    - **Scikit-learn**: For Machine Learning classifiers (SVM, KNN, Random Forest).
- **Security**:
    - **Cryptography (Fernet)**: For encrypting sensitive user data.
    - **UUID**: For anonymizing feature storage.
- **Web Framework**:
    - **Flask**: For serving the web application and APIs.
- **Frontend**:
    - **HTML5 / JavaScript**: For the user interface and webcam integration.

## 🏗️ Architecture

1.  **Input**: Webcam feed (Face) or Image Upload (Fingerprint).
2.  **Preprocessing**:
    - **Face**: Alignment, cropping, resizing.
    - **Fingerprint**: Histogram equalization, resizing, HOG feature extraction.
3.  **Feature Extraction**: Converting images into numerical feature vectors.
4.  **Secure Storage**:
    - Features saved with `UUID.ext` filenames.
    - Metadata saved as `UUID | Encrypted(Name|Country)`.
5.  **Matching**:
    - **Math Matcher**: Euclidean distance / Similarity metrics.
    - **ML Matcher**: Trained classifier prediction.
6.  **Decryption**: If a match is found, the system looks up the UUID and decrypts the user's name for display.
