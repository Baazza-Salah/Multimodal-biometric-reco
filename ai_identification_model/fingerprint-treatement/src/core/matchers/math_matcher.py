# Enhanced Math Matcher - Multi-Template Support
# Supports multiple enrollment images per person with score aggregation

import numpy as np
import re
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import StandardScaler
from collections import defaultdict

class MathMatcher:
    """
    Advanced biometric matcher with multi-template support.
    
    Handles multiple templates per person (e.g., finger1_1, finger1_2).
    Aggregates scores across all templates of each person for robust matching.
    """
    
    def __init__(self):
        self.template_features = None
        self.template_labels = []  # Individual template labels
        self.person_names = []  # Unique person names
        self.person_to_templates = {}  # person -> list of template indices
        self.scaler = StandardScaler()
        self._is_fitted = False
        self.feature_weights = None
        
    def _extract_person_name(self, label: str) -> str:
        """
        Extract person name from label.
        Examples: user1 -> user, 1__M_Left... -> 1
        """
        # Handle fingerprint format
        if '__' in label:
            return label.split('__')[0]
            
        # Remove trailing numbers
        base_name = re.sub(r'\d+$', '', label)
        return base_name if base_name else label
        
    def _compute_feature_weights(self, features: np.ndarray):
        """Compute importance weights based on feature variance."""
        variances = np.var(features, axis=0)
        weights = variances / (np.mean(variances) + 1e-8)
        weights = np.clip(weights, 0.1, 10.0)
        return weights
        
    def fit(self, features: np.ndarray, labels: np.ndarray):
        """
        Store templates and group by person.
        
        Args:
            features: (n_templates, n_features) array
            labels: (n_templates,) array of template identifiers
        """
        self.template_labels = labels.tolist()
        
        # Group templates by person
        person_templates = defaultdict(list)
        for idx, label in enumerate(self.template_labels):
            person = self._extract_person_name(label)
            person_templates[person].append(idx)
        
        self.person_names = sorted(person_templates.keys())
        self.person_to_templates = dict(person_templates)
        
        # Standardize features
        self.scaler.fit(features)
        self.template_features = self.scaler.transform(features)
        
        # Compute feature weights
        self.feature_weights = self._compute_feature_weights(self.template_features)
        
        self._is_fitted = True
        
        print(f"Enrolled {len(self.person_names)} persons with {len(self.template_labels)} templates total")
        for person in self.person_names:
            n_templates = len(self.person_to_templates[person])
            print(f"  - {person}: {n_templates} template(s)")
        
    def _weighted_euclidean_distance(self, probe: np.ndarray, templates: np.ndarray):
        """Calculate weighted Euclidean distance."""
        diff = probe - templates
        weighted_diff = diff * np.sqrt(self.feature_weights)
        distances = np.linalg.norm(weighted_diff, axis=1)
        return distances
        
    def predict(self, probe_features: np.ndarray):
        """
        Identify person from probe features using multi-template matching.
        
        Returns:
            List of person-level results with aggregated scores
        """
        if not self._is_fitted:
            raise RuntimeError("Matcher not fitted. Call fit() first.")
        
        # Reshape if needed
        if probe_features.ndim == 1:
            probe_features = probe_features.reshape(1, -1)
        
        # Standardize probe
        probe_scaled = self.scaler.transform(probe_features)[0]
        
        # Calculate distances to all templates
        all_distances = self._weighted_euclidean_distance(
            probe_scaled, 
            self.template_features
        )
        
        # Aggregate scores per person
        person_results = []
        
        for person in self.person_names:
            template_indices = self.person_to_templates[person]
            person_distances = all_distances[template_indices]
            
            # Aggregation strategies:
            # 1. Min distance (best match among all templates)
            min_dist = np.min(person_distances)
            # 2. Mean distance (average across templates)
            mean_dist = np.mean(person_distances)
            # 3. Median distance
            median_dist = np.median(person_distances)
            
            # Use minimum distance (most generous, suitable for multi-pose)
            best_dist = min_dist
            
            # Convert to similarity
            person_results.append({
                'person': person,
                'min_distance': float(min_dist),
                'mean_distance': float(mean_dist),
                'median_distance': float(median_dist),
                'best_distance': float(best_dist),
                'n_templates': len(template_indices)
            })
        
        # Sort by best distance
        person_results.sort(key=lambda x: x['best_distance'])
        
        # Calculate scores
        distances = np.array([r['best_distance'] for r in person_results])
        
        # RBF kernel for similarity
        median_dist = np.median(distances)
        gamma = 1.0 / (2 * median_dist**2 + 1e-8)
        similarities = np.exp(-gamma * distances**2)
        
        # Relative scores (sum to 100%)
        relative_scores = (similarities / np.sum(similarities)) * 100
        
        # Absolute scores (0-100 based on max distance)
        max_dist = np.max(distances)
        if max_dist > 0:
            absolute_scores = (1 - (distances / max_dist)) * 100
        else:
            absolute_scores = np.array([100.0] + [0.0] * (len(distances) - 1))
        
        # Add scores to results
        for i, result in enumerate(person_results):
            result['relative_score'] = float(relative_scores[i])
            result['absolute_score'] = float(absolute_scores[i])
            result['similarity'] = float(similarities[i])
        
        return person_results
    
    def identify(self, probe_features: np.ndarray, 
                 min_gap: float = 10.0, 
                 min_absolute: float = 60.0):
        """
        Identify person with adaptive thresholding.
        
        Args:
            probe_features: Feature vector
            min_gap: Minimum gap between best and second best
            min_absolute: Minimum absolute score for acceptance
            
        Returns:
            (person_name or "Unknown", confidence_dict, all_results)
        """
        results = self.predict(probe_features)
        
        best = results[0]
        second_best = results[1] if len(results) > 1 else None
        
        # Calculate gap
        if second_best:
            gap = best['relative_score'] - second_best['relative_score']
        else:
            gap = 100.0
        
        # Decision criteria
        accept = (best['absolute_score'] >= min_absolute and gap >= min_gap)
        
        if accept:
            identified = best['person']
            confidence = {
                'absolute': best['absolute_score'],
                'relative': best['relative_score'],
                'gap': gap,
                'distance': best['best_distance'],
                'n_templates': best['n_templates']
            }
        else:
            identified = "Unknown"
            confidence = {
                'absolute': best['absolute_score'],
                'relative': best['relative_score'],
                'gap': gap,
                'distance': best['best_distance'],
                'n_templates': best['n_templates'],
                'reason': f"Low absolute score ({best['absolute_score']:.1f}%)" if best['absolute_score'] < min_absolute else f"Small gap ({gap:.1f}%)"
            }
        
        return identified, confidence, results


def load_database(csv_path):
    """Load database from CSV file."""
    import pandas as pd
    from pathlib import Path
    
    df = pd.read_csv(csv_path)
    # First column is filename (label), rest are features
    labels = df['filename'].str.replace(r'\.(png|jpg|jpeg|bmp|tif)$', '', regex=True).values
    features = df.iloc[:, 1:].values.astype('float32')
    return labels, features
