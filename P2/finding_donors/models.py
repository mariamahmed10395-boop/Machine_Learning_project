import random
import numpy as np
from typing import Dict, List, Optional, Tuple, Any

def sigmoid(z: float) -> float:
    """Sigmoid activation function."""
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

class SimpleRandomForest:
    """Simplified Random Forest classifier."""
    
    def __init__(self, n_estimators: int = 100, max_depth: int = 5):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.trees: List[Dict] = []
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train the random forest."""
        self.trees = []
        n_features = X.shape[1]
        n_samples = X.shape[0]
        
        for _ in range(self.n_estimators):
            # Bootstrap sample
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            X_sample = X[indices]
            y_sample = y[indices]
            
            # Random feature
            feature = random.randint(0, n_features - 1)
            threshold = np.mean(X_sample[:, feature])
            
            # Simple split
            left_mask = X_sample[:, feature] <= threshold
            right_mask = ~left_mask
            
            left_labels = y_sample[left_mask]
            right_labels = y_sample[right_mask]
            
            left_pred = int(np.mean(left_labels) > 0.5) if len(left_labels) > 0 else 0
            right_pred = int(np.mean(right_labels) > 0.5) if len(right_labels) > 0 else 0
            
            self.trees.append({
                'feature': feature,
                'threshold': threshold,
                'left_pred': left_pred,
                'right_pred': right_pred
            })
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        predictions = []
        for x in X:
            votes = []
            for tree in self.trees:
                pred = tree['left_pred'] if x[tree['feature']] <= tree['threshold'] else tree['right_pred']
                votes.append(pred)
            predictions.append(1 if np.mean(votes) > 0.5 else 0)
        return np.array(predictions)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities."""
        probabilities = []
        for x in X:
            votes = []
            for tree in self.trees:
                pred = tree['left_pred'] if x[tree['feature']] <= tree['threshold'] else tree['right_pred']
                votes.append(pred)
            prob = np.mean(votes)
            probabilities.append([1 - prob, prob])
        return np.array(probabilities)
    
    def feature_importances(self, n_features: int = 5) -> np.ndarray:
        """Get feature importances."""
        counts = np.zeros(n_features)
        for tree in self.trees:
            counts[tree['feature']] += 1
        return counts / len(self.trees) if len(self.trees) > 0 else counts


class SimpleLogisticRegression:
    """Simplified Logistic Regression classifier."""
    
    def __init__(self, max_iter: int = 1000, learning_rate: float = 0.1):
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.weights: Optional[np.ndarray] = None
        self.bias: float = 0.0
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train using gradient descent."""
        n_features = X.shape[1]
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        
        for _ in range(self.max_iter):
            # Forward pass
            z = np.dot(X, self.weights) + self.bias
            predictions = sigmoid(z)
            
            # Gradients
            dw = np.dot(X.T, (predictions - y)) / len(y)
            db = np.mean(predictions - y)
            
            # Update
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        proba = self.predict_proba(X)[:, 1]
        return (proba > 0.5).astype(int)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities."""
        z = np.dot(X, self.weights) + self.bias
        prob = sigmoid(z)
        return np.column_stack([1 - prob, prob])
    
    def coef_(self) -> np.ndarray:
        """Get coefficients."""
        return self.weights if self.weights is not None else np.array([])


class SimpleGradientBoosting:
    """Simplified Gradient Boosting classifier."""
    
    def __init__(self, n_estimators: int = 100, max_depth: int = 5, learning_rate: float = 0.1):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.trees: List[Dict] = []
        self.initial_prediction: float = 0.0
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train using gradient boosting."""
        self.trees = []
        self.initial_prediction = np.mean(y)
        residuals = y - self.initial_prediction
        
        for _ in range(self.n_estimators):
            feature = random.randint(0, X.shape[1] - 1)
            threshold = np.mean(X[:, feature])
            
            left_mask = X[:, feature] <= threshold
            right_mask = ~left_mask
            
            left_residuals = residuals[left_mask]
            right_residuals = residuals[right_mask]
            
            left_value = np.mean(left_residuals) * self.learning_rate if len(left_residuals) > 0 else 0
            right_value = np.mean(right_residuals) * self.learning_rate if len(right_residuals) > 0 else 0
            
            self.trees.append({
                'feature': feature,
                'threshold': threshold,
                'left_value': left_value,
                'right_value': right_value
            })
            
            # Update residuals
            for i, x in enumerate(X):
                tree = self.trees[-1]
                update = tree['left_value'] if x[tree['feature']] <= tree['threshold'] else tree['right_value']
                residuals[i] -= update
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        proba = self.predict_proba(X)[:, 1]
        return (proba > 0.5).astype(int)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities."""
        probabilities = []
        for x in X:
            pred = self.initial_prediction
            for tree in self.trees:
                pred += tree['left_value'] if x[tree['feature']] <= tree['threshold'] else tree['right_value']
            prob = np.clip(pred, 0, 1)
            probabilities.append([1 - prob, prob])
        return np.array(probabilities)
    
    def feature_importances(self, n_features: int = 5) -> np.ndarray:
        """Get feature importances."""
        counts = np.zeros(n_features)
        for tree in self.trees:
            counts[tree['feature']] += 1
        return counts / len(self.trees) if len(self.trees) > 0 else counts
