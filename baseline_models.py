"""
Baseline Models for Glucose Onset Detection
==========================================

Minimal baseline model collection with only logistic regression
to avoid dependency issues.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


class LogisticRegressionBaseline:
    """Simple logistic regression with class weighting."""
    
    def __init__(self, class_weights=None):
        if class_weights is None:
            class_weights = {0: 1.0, 1: 5.0, 2: 10.0}  # Same as CNN
        self.model = LogisticRegression(
            class_weight=class_weights,
            multi_class='ovr',
            max_iter=1000,
            random_state=42
        )
        self.scaler = StandardScaler()
    
    def fit(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        
    def predict_proba(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
    
    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)


# Removed all complex models to avoid dependency issues


class RuleBasedBaseline:
    """Simple rule-based baseline using glucose thresholds and trends."""
    
    def __init__(self, 
                 low_threshold=70,      # mg/dL threshold for low glucose
                 drop_threshold=30,     # mg/dL drop indicating trend
                 variance_threshold=200, # High variance indicating compression
                 compression_glucose_range=(30, 90)):  # Range where compression is likely
        
        self.low_threshold = low_threshold
        self.drop_threshold = drop_threshold  
        self.variance_threshold = variance_threshold
        self.compression_range = compression_glucose_range
    
    def fit(self, X, y):
        """Rule-based model doesn't need training, but we can tune thresholds."""
        # Could implement threshold optimization here
        pass
    
    def predict_proba(self, X):
        """
        X should contain: [glucose, d1, d5, accel, var15, pct10]
        """
        n_samples = X.shape[0]
        probs = np.zeros((n_samples, 3))
        
        for i in range(n_samples):
            glucose, d1, d5, accel, var15, pct10 = X[i]
            
            # Default to "none" class
            prob_none = 0.7
            prob_compression = 0.15  
            prob_regular = 0.15
            
            # Rule 1: High variance suggests compression
            if var15 > self.variance_threshold:
                if (self.compression_range[0] <= glucose <= self.compression_range[1]):
                    prob_compression = 0.6
                    prob_none = 0.25
                    prob_regular = 0.15
            
            # Rule 2: Low glucose with steady drop suggests regular low
            elif glucose < self.low_threshold and d5 < -10:
                prob_regular = 0.7
                prob_none = 0.2
                prob_compression = 0.1
            
            # Rule 3: Rapid drops (could be either)
            elif abs(pct10) > 0.2 or d5 < -self.drop_threshold:
                if var15 > self.variance_threshold / 2:
                    prob_compression = 0.45
                    prob_regular = 0.35
                    prob_none = 0.2
                else:
                    prob_regular = 0.5
                    prob_compression = 0.2
                    prob_none = 0.3
            
            probs[i] = [prob_none, prob_compression, prob_regular]
        
        return probs
    
    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)


def get_all_baselines():
    """Get dictionary of minimal baseline models."""
    return {
        'logistic_regression': LogisticRegressionBaseline(),
    }


def get_pytorch_baselines():
    """Get PyTorch-based baseline models (empty for minimal version)."""
    return {}


def get_all_baselines_wrapped():
    """Get all baselines with proper wrapping for class weights (minimal version)."""
    return get_all_baselines()