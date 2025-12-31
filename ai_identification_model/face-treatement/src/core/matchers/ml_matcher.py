# ML Matcher - Machine Learning Based Biometric Identification
# Trains classifiers (KNN, SVM, Random Forest) with hyperparameter tuning

import numpy as np
import pickle
import re
from pathlib import Path
from collections import defaultdict
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, classification_report

class MLMatcher:
    """
    ML-based biometric matcher with multiple algorithms.
    
    Supports:
    - KNN (K-Nearest Neighbors)
    - SVM (Support Vector Machine)
    - Random Forest
    
    Includes hyperparameter tuning and model persistence.
    """
    
    def __init__(self, algorithm='auto'):
        """
        Args:
            algorithm: 'knn', 'svm', 'random_forest', or 'auto' (selects best)
        """
        self.algorithm = algorithm
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self._is_fitted = False
        self.best_params = {}
        self.cv_score = 0.0
        
    def _extract_person_name(self, label: str) -> str:
        """Extract person name from label (remove trailing numbers)."""
        base_name = re.sub(r'\d+$', '', label)
        return base_name if base_name else label
    
    def _prepare_data(self, features, labels):
        """Group multiple templates per person into single labels."""
        # Extract person names
        person_labels = np.array([self._extract_person_name(l) for l in labels])
        
        # Encode labels to integers
        encoded_labels = self.label_encoder.fit_transform(person_labels)
        
        # Standardize features
        scaled_features = self.scaler.fit_transform(features)
        
        return scaled_features, encoded_labels, person_labels
    
    def _train_knn(self, X, y):
        """Train KNN with hyperparameter tuning."""
        print("Training KNN classifier...")
        param_grid = {
            'n_neighbors': [1, 3, 5, 7, 9],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan', 'minkowski']
        }
        
        knn = KNeighborsClassifier()
        grid_search = GridSearchCV(knn, param_grid, cv=min(3, len(np.unique(y))), 
                                   scoring='accuracy', n_jobs=-1)
        grid_search.fit(X, y)
        
        self.best_params['knn'] = grid_search.best_params_
        return grid_search.best_estimator_, grid_search.best_score_
    
    def _train_svm(self, X, y):
        """Train SVM with hyperparameter tuning."""
        print("Training SVM classifier...")
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.001, 0.01],
            'kernel': ['rbf', 'linear']
        }
        
        svm = SVC(probability=True)  # Enable probability estimates
        grid_search = GridSearchCV(svm, param_grid, cv=min(3, len(np.unique(y))),
                                   scoring='accuracy', n_jobs=-1)
        grid_search.fit(X, y)
        
        self.best_params['svm'] = grid_search.best_params_
        return grid_search.best_estimator_, grid_search.best_score_
    
    def _train_random_forest(self, X, y):
        """Train Random Forest with hyperparameter tuning."""
        print("Training Random Forest classifier...")
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        rf = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(rf, param_grid, cv=min(3, len(np.unique(y))),
                                   scoring='accuracy', n_jobs=-1)
        grid_search.fit(X, y)
        
        self.best_params['rf'] = grid_search.best_params_
        return grid_search.best_estimator_, grid_search.best_score_
    
    def fit(self, features: np.ndarray, labels: np.ndarray):
        """
        Train the ML model.
        
        Args:
            features: (n_samples, n_features) array
            labels: (n_samples,) array of identifiers
        """
        # Prepare data
        X, y, person_names = self._prepare_data(features, labels)
        
        print(f"\nTraining ML model on {len(X)} samples, {len(np.unique(y))} persons")
        print(f"Persons: {', '.join(self.label_encoder.classes_)}")
        
        if self.algorithm == 'auto':
            # Train all and select best
            models = {}
            scores = {}
            
            models['knn'], scores['knn'] = self._train_knn(X, y)
            models['svm'], scores['svm'] = self._train_svm(X, y)
            models['rf'], scores['rf'] = self._train_random_forest(X, y)
            
            # Select best
            best_algo = max(scores, key=scores.get)
            self.model = models[best_algo]
            self.cv_score = scores[best_algo]
            self.algorithm = best_algo
            
            self._is_fitted = True
            
            return {
                'best_algo': best_algo,
                'best_score': self.cv_score,
                'all_scores': scores,
                'best_params': self.best_params[best_algo]
            }
            
        elif self.algorithm == 'knn':
            self.model, self.cv_score = self._train_knn(X, y)
        elif self.algorithm == 'svm':
            self.model, self.cv_score = self._train_svm(X, y)
        elif self.algorithm == 'random_forest':
            self.model, self.cv_score = self._train_random_forest(X, y)
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")
        
        self._is_fitted = True
        return {
            'best_algo': self.algorithm,
            'best_score': self.cv_score,
            'all_scores': {self.algorithm: self.cv_score},
            'best_params': self.best_params.get(self.algorithm, {})
        }
        
    def predict(self, probe_features: np.ndarray):
        """
        Predict identity from probe features.
        
        Returns:
            List of results with confidence scores
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        # Reshape if needed
        if probe_features.ndim == 1:
            probe_features = probe_features.reshape(1, -1)
        
        # Scale features
        X_scaled = self.scaler.transform(probe_features)
        
        # Get predictions and probabilities
        prediction = self.model.predict(X_scaled)[0]
        probabilities = self.model.predict_proba(X_scaled)[0]
        
        # Build results
        results = []
        for idx, person in enumerate(self.label_encoder.classes_):
            results.append({
                'person': person,
                'probability': float(probabilities[idx]),
                'confidence': float(probabilities[idx] * 100)
            })
        
        # Sort by probability
        results.sort(key=lambda x: x['probability'], reverse=True)
        
        return results
    
    def identify(self, probe_features: np.ndarray, min_confidence: float = 50.0):
        """
        Identify person with confidence threshold.
        
        Args:
            probe_features: Feature vector
            min_confidence: Minimum confidence percentage (0-100)
            
        Returns:
            (person_name or "Unknown", confidence_dict, all_results)
        """
        results = self.predict(probe_features)
        
        best = results[0]
        second_best = results[1] if len(results) > 1 else None
        
        # Calculate gap
        if second_best:
            gap = best['confidence'] - second_best['confidence']
        else:
            gap = 100.0
        
        # Decision
        if best['confidence'] >= min_confidence:
            identified = best['person']
            confidence_dict = {
                'confidence': best['confidence'],
                'probability': best['probability'],
                'gap': gap,
                'algorithm': self.algorithm,
                'cv_score': self.cv_score * 100
            }
        else:
            identified = "Unknown"
            confidence_dict = {
                'confidence': best['confidence'],
                'probability': best['probability'],
                'gap': gap,
                'algorithm': self.algorithm,
                'cv_score': self.cv_score * 100,
                'reason': f"Low confidence ({best['confidence']:.1f}%)"
            }
        
        return identified, confidence_dict, results
    
    def save(self, filepath: Path):
        """Save trained model to file."""
        if not self._is_fitted:
            raise RuntimeError("Cannot save unfitted model")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'algorithm': self.algorithm,
            'best_params': self.best_params,
            'cv_score': self.cv_score
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filepath}")
    
    def load(self, filepath: Path):
        """Load trained model from file."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.label_encoder = model_data['label_encoder']
        self.algorithm = model_data['algorithm']
        self.best_params = model_data['best_params']
        self.cv_score = model_data['cv_score']
        self._is_fitted = True
        
        print(f"Model loaded from {filepath}")
        print(f"Algorithm: {self.algorithm}")
        print(f"CV Score: {self.cv_score:.2%}")


def load_database(csv_path):
    """Load database from CSV file."""
    import pandas as pd
    
    df = pd.read_csv(csv_path)
    labels = df['filename'].str.replace(r'\.(png|jpg|jpeg)$', '', regex=True).values
    features = df.iloc[:, 1:].values.astype('float32')
    return labels, features
