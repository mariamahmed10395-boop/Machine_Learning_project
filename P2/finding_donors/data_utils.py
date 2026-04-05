import random
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from constants import (
    WORKCLASS_OPTS, EDUCATION_OPTS, MARITAL_OPTS, 
    OCCUPATION_OPTS, RELATIONSHIP_OPTS, RACE_OPTS, 
    SEX_OPTS, COUNTRY_OPTS
)

def generate_synthetic_data(n_samples: int = 32561) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic census-like data with 13 features."""
    np.random.seed(42)
    random.seed(42)
    
    features = []
    labels = []
    
    for _ in range(n_samples):
        # 1. الميزات الرقمية (القديمة)
        age = random.randint(17, 90)
        education_num = random.randint(1, 16)
        capital_gain = 0 if random.random() < 0.9 else random.randint(1000, 99999)
        capital_loss = 0 if random.random() < 0.95 else random.randint(100, 4356)
        hours_per_week = random.randint(20, 80)
        
        # 2. الميزات الجديدة (قيم عشوائية لتمثيل الاختيارات)
        work_idx = random.randint(0, len(WORKCLASS_OPTS) - 1)
        edu_idx = min(education_num - 1, len(EDUCATION_OPTS) - 1)
        mar_idx = random.randint(0, len(MARITAL_OPTS) - 1)
        occ_idx = random.randint(0, len(OCCUPATION_OPTS) - 1)
        rel_idx = random.randint(0, len(RELATIONSHIP_OPTS) - 1)
        race_idx = random.randint(0, len(RACE_OPTS) - 1)
        sex_idx = random.randint(0, len(SEX_OPTS) - 1)
        nat_idx = random.randint(0, len(COUNTRY_OPTS) - 1)
        
        features.append([
            age, education_num, capital_gain, capital_loss, hours_per_week,
            work_idx, edu_idx, mar_idx, occ_idx, rel_idx, race_idx, sex_idx, nat_idx
        ])
        
        # تحديد النتيجة (Label) بناءً على نمط منطقي
        score = 0
        if education_num >= 13: score += 2
        if 30 <= age <= 60: score += 1
        if capital_gain > 5000: score += 3
        if hours_per_week >= 40: score += 1
        if sex_idx == 1: score += 0.5  # إضافة وزن بسيط كمثال
        
        is_rich = score >= 4 or random.random() < 0.24
        labels.append(1 if is_rich else 0)
    
    return np.array(features, dtype=float), np.array(labels)


def preprocess_features(X: np.ndarray) -> Tuple[np.ndarray, Dict]:
    """Preprocess features: log transform and min-max scaling."""
    X_transformed = X.copy()
    
    # Log transform capital-gain (index 2) and capital-loss (index 3)
    X_transformed[:, 2] = np.log1p(X_transformed[:, 2])
    X_transformed[:, 3] = np.log1p(X_transformed[:, 3])
    
    # Min-max scaling
    mins = X_transformed.min(axis=0)
    maxs = X_transformed.max(axis=0)
    ranges = maxs - mins
    ranges[ranges == 0] = 1  # Avoid division by zero
    
    X_scaled = (X_transformed - mins) / ranges
    
    scaler_params = {'mins': mins, 'maxs': maxs, 'ranges': ranges}
    
    return X_scaled, scaler_params


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate accuracy and F0.5 score."""
    accuracy = np.mean(y_true == y_pred)
    
    # Calculate F0.5 score
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    beta = 0.5
    beta_sq = beta ** 2
    
    if precision + recall == 0:
        f05 = 0.0
    else:
        f05 = (1 + beta_sq) * (precision * recall) / (beta_sq * precision + recall)
    
    return {'accuracy': accuracy, 'f05': f05, 'precision': precision, 'recall': recall}


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Calculate confusion matrix."""
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tp = np.sum((y_true == 1) & (y_pred == 1))
    return np.array([[tn, fp], [fn, tp]])
