"""
CharityML — Finding Donors GUI
================================
Modern Tkinter GUI for donor prediction using machine learning.
Implements Random Forest, Logistic Regression, and Gradient Boosting.

Run: python charityml_donor_predictor.py
"""

import os
import random
import threading
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from tkinter.scrolledtext import ScrolledText
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

# Try to import matplotlib for charts
try:
    import matplotlib
    matplotlib.use('TkAgg')
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Matplotlib not available. Charts will be disabled.")

# =============================================================================
# CONSTANTS
# =============================================================================

# Feature options (from the original census dataset)
WORKCLASS_OPTS = [
    'Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 
    'Local-gov', 'State-gov', 'Without-pay', 'Never-worked'
]

EDUCATION_OPTS = [
    'Preschool', '1st-4th', '5th-6th', '7th-8th', '9th', '10th', 
    '11th', '12th', 'HS-grad', 'Some-college', 'Assoc-voc', 
    'Assoc-acdm', 'Bachelors', 'Masters', 'Prof-school', 'Doctorate'
]

MARITAL_OPTS = [
    'Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 
    'Widowed', 'Married-spouse-absent', 'Married-AF-spouse'
]

OCCUPATION_OPTS = [
    'Tech-support', 'Craft-repair', 'Other-service', 'Sales', 
    'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners', 
    'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing', 
    'Transport-moving', 'Priv-house-serv', 'Protective-serv', 'Armed-Forces'
]

RELATIONSHIP_OPTS = ['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried']
RACE_OPTS = ['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other']
SEX_OPTS = ['Male', 'Female']

COUNTRY_OPTS = [
    'United-States', 'Cambodia', 'England', 'Puerto-Rico', 'Canada', 'Germany',
    'Outlying-US(Guam-USVI-etc)', 'India', 'Japan', 'Greece', 'South', 'China',
    'Cuba', 'Iran', 'Honduras', 'Philippines', 'Italy', 'Poland', 'Jamaica',
    'Vietnam', 'Mexico', 'Portugal', 'Ireland', 'France', 'Dominican-Republic',
    'Laos', 'Ecuador', 'Taiwan', 'Haiti', 'Columbia', 'Hungary', 'Guatemala',
    'Nicaragua', 'Scotland', 'Thailand', 'Yugoslavia', 'El-Salvador',
    'Trinadad&Tobago', 'Peru', 'Hong', 'Holand-Netherlands'
]

# Color palette (Dark Navy + Teal theme)
COLORS = {
    'bg': '#062C22',          
    'surface': '#0A3D31',     
    'card': '#114D40',     
    'border': '#D4AF37', 
    'accent': '#F1C40F',      
    'accent_light': '#FDE68A',
    'green': '#2ECC71',
    'red': '#E74C3C',
    'purple': '#9B59B6',
    'orange': '#E67E22',
    'text': '#ECF0F1',        # أبيض مائل للرمادي
    'text_secondary': '#BDC3C7',
    'white': '#FFFFFF'
}

# Default feature values
DEFAULT_FEATURES = {
    'age': 38,
    'education-num': 10,
    'capital-gain': 0,
    'capital-loss': 0,
    'hours-per-week': 40,
    'workclass': 'Private',
    'education_level': 'Bachelors',
    'marital-status': 'Never-married',
    'occupation': 'Adm-clerical',
    'relationship': 'Not-in-family',
    'race': 'White',
    'sex': 'Male',
    'native-country': 'United-States'
}

TYPICAL_DONOR = {
    'age': 45,
    'education-num': 13,
    'capital-gain': 7688,
    'capital-loss': 0,
    'hours-per-week': 50,
    'workclass': 'Private',
    'education_level': 'Bachelors',
    'marital-status': 'Married-civ-spouse',
    'occupation': 'Exec-managerial',
    'relationship': 'Husband',
    'race': 'White',
    'sex': 'Male',
    'native-country': 'United-States'
}

# =============================================================================
# MACHINE LEARNING IMPLEMENTATIONS
# =============================================================================

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


# =============================================================================
# DATA GENERATION AND PREPROCESSING
# =============================================================================

def generate_synthetic_data(n_samples: int = 32561) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic census-like data with 9 features."""
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
        
        # 2. الميزات الجديدة (قيم عشوائية بين 0 و 1 لتمثيل الاختيارات)
        workclass_val = random.random()
        edu_level_val = random.random()
        marital_val = random.random()
        sex_val = random.choice([0, 1])  # 0 للذكور، 1 للإناث
        
        # إضافة الـ 9 ميزات بالترتيب الصحيح
        features.append([
            age, education_num, capital_gain, capital_loss, hours_per_week,
            workclass_val, edu_level_val, marital_val, sex_val
        ])
        
        # تحديد النتيجة (Label) بناءً على نمط منطقي
        score = 0
        if education_num >= 13: score += 2
        if 30 <= age <= 60: score += 1
        if capital_gain > 5000: score += 3
        if hours_per_week >= 40: score += 1
        if sex_val == 1: score += 0.5  # إضافة وزن بسيط للجنس كمثال
        
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


# =============================================================================
# MAIN GUI APPLICATION
# =============================================================================

class CharityMLApp:
    """Main CharityML Donor Predictor Application."""
    
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("CharityML — Donor Finding Predictor")
        self.root.geometry("1400x900")
        self.root.minsize(1200, 800)
        self.root.configure(bg=COLORS['bg'])
        
        # State variables
        self.features = DEFAULT_FEATURES.copy()
        self.dataset_loaded = False
        self.dataset_info: Optional[Dict] = None
        self.scaler_params: Optional[Dict] = None
        self.X_train: Optional[np.ndarray] = None
        self.X_test: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.y_test: Optional[np.ndarray] = None
        
        # Models
        self.models: Dict[str, Any] = {
            'Random Forest': None,
            'Logistic Regression': None,
            'Gradient Boosting': None
        }
        self.results: Dict[str, Optional[Dict]] = {
            'Random Forest': None,
            'Logistic Regression': None,
            'Gradient Boosting': None
        }
        
        # Model configuration
        self.model_config = {
            'n_estimators': tk.IntVar(value=100),
            'max_depth': tk.IntVar(value=5),
            'learning_rate': tk.DoubleVar(value=0.1),
            'test_size': tk.DoubleVar(value=0.2)
        }
        
        self.selected_model = tk.StringVar(value='Gradient Boosting')
        self.is_training = False
        
        # Build UI
        self._setup_styles()
        self._build_ui()
    
    def _setup_styles(self):
        """Configure ttk styles."""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure colors
        style.configure('TFrame', background=COLORS['surface'])
        style.configure('TLabel', background=COLORS['surface'], foreground=COLORS['text'])
        style.configure('TButton', background=COLORS['card'], foreground=COLORS['text'])
        style.configure('TCombobox', fieldbackground=COLORS['card'], foreground=COLORS['text'])
        style.configure('Horizontal.TProgressbar', 
                       troughcolor=COLORS['card'], 
                       background=COLORS['accent'])
        
        # Custom styles
        style.configure('Card.TFrame', background=COLORS['card'])
        style.configure('Title.TLabel', 
                       background=COLORS['surface'], 
                       foreground=COLORS['accent'],
                       font=('Space Grotesk', 11, 'bold'))
    
    def _build_ui(self):
        """Build the main user interface."""
        # Header
        self._build_header()
        
        # Main content area
        self.main_frame = tk.Frame(self.root, bg=COLORS['bg'])
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Three-column layout
        self.main_frame.columnconfigure(0, weight=3, uniform='col')
        self.main_frame.columnconfigure(1, weight=3, uniform='col')
        self.main_frame.columnconfigure(2, weight=4, uniform='col')
        self.main_frame.rowconfigure(0, weight=1)
        
        # Build panels
        self._build_left_panel()
        self._build_center_panel()
        self._build_right_panel()
        
        # Status bar
        self._build_status_bar()
    
    def _build_header(self):
        """Build the header bar."""
        header = tk.Frame(self.root, bg=COLORS['surface'], height=60)
        header.pack(fill=tk.X)
        header.pack_propagate(False)
        
        # Logo
        tk.Label(header, 
                text="CharityML", 
                bg=COLORS['surface'], 
                fg=COLORS['accent'],
                font=('Space Grotesk', 18, 'bold')).pack(side=tk.LEFT, padx=20, pady=15)
        
        # Tagline
        tk.Label(header,
                text="Finding Donors Predictor",
                bg=COLORS['surface'],
                fg=COLORS['text_secondary'],
                font=('Inter', 11)).pack(side=tk.LEFT, padx=5)
        
        # Models indicator
        models_text = "Random Forest  ·  Logistic Regression  ·  Gradient Boosting"
        tk.Label(header,
                text=models_text,
                bg=COLORS['surface'],
                fg=COLORS['text_secondary'],
                font=('IBM Plex Mono', 9)).pack(side=tk.RIGHT, padx=20)
    
    def _build_left_panel(self):
        """Build the left panel - Dataset & Model Config."""
        panel = tk.Frame(self.main_frame, bg=COLORS['surface'], 
                        highlightbackground=COLORS['border'], 
                        highlightthickness=1)
        panel.grid(row=0, column=0, sticky='nsew', padx=(0, 10), pady=5)
        
        # Dataset Section
        tk.Label(panel, 
                text="📊  DATASET", 
                bg=COLORS['surface'], 
                fg=COLORS['accent'],
                font=('IBM Plex Mono', 10, 'bold')).pack(anchor='w', padx=15, pady=(15, 5))
        
        # Dataset info display
        self.dataset_frame = tk.Frame(panel, bg=COLORS['card'], padx=10, pady=10)
        self.dataset_frame.pack(fill=tk.X, padx=15, pady=5)
        
        self.dataset_label = tk.Label(self.dataset_frame,
                                     text="No dataset loaded",
                                     bg=COLORS['card'],
                                     fg=COLORS['text_secondary'],
                                     font=('Inter', 10))
        self.dataset_label.pack()
        
        # Load buttons
        btn_frame = tk.Frame(panel, bg=COLORS['surface'])
        btn_frame.pack(fill=tk.X, padx=15, pady=5)
        
        self._create_button(btn_frame, "Browse", self._browse_file, 
                           COLORS['card'], COLORS['text']).pack(side=tk.LEFT, padx=(0, 5))
        self._create_button(btn_frame, "▶  Load Sample Data", self._load_sample,
                           COLORS['accent'], COLORS['bg'], font=('Inter', 10, 'bold')).pack(side=tk.LEFT)
        
        # Separator
        tk.Frame(panel, bg=COLORS['border'], height=1).pack(fill=tk.X, padx=15, pady=15)
        
        # Model Selection Section
        tk.Label(panel,
                text="🤖  MODEL CONFIGURATION",
                bg=COLORS['surface'],
                fg=COLORS['accent'],
                font=('IBM Plex Mono', 10, 'bold')).pack(anchor='w', padx=15, pady=(5, 10))
        
        # Model selection
        model_frame = tk.Frame(panel, bg=COLORS['surface'])
        model_frame.pack(fill=tk.X, padx=15, pady=5)
        
        models = [
            ('Random Forest', COLORS['green'], '🌲'),
            ('Logistic Regression', COLORS['purple'], '📈'),
            ('Gradient Boosting', COLORS['orange'], '🚀')
        ]
        
        for name, color, icon in models:
            self._build_model_radio(model_frame, name, color, icon)
        
        # Hyperparameters
        tk.Label(panel,
                text="Hyperparameters",
                bg=COLORS['surface'],
                fg=COLORS['text_secondary'],
                font=('Inter', 10, 'bold')).pack(anchor='w', padx=15, pady=(15, 10))
        
        params_frame = tk.Frame(panel, bg=COLORS['surface'])
        params_frame.pack(fill=tk.X, padx=15, pady=5)
        
        self._build_param_spinbox(params_frame, "N Estimators", self.model_config['n_estimators'], 10, 500, 1)
        self._build_param_spinbox(params_frame, "Max Depth", self.model_config['max_depth'], 1, 20, 1)
        self._build_param_spinbox(params_frame, "Learning Rate", self.model_config['learning_rate'], 0.01, 1.0, 0.01)
        self._build_param_spinbox(params_frame, "Test Size", self.model_config['test_size'], 0.1, 0.4, 0.05)
        
        # Training buttons
        btn_frame2 = tk.Frame(panel, bg=COLORS['surface'])
        btn_frame2.pack(fill=tk.X, padx=15, pady=15)
        
        self._create_button(btn_frame2, "🎯  Train Model", self._train_selected,
                           COLORS['accent'], COLORS['bg'], font=('Inter', 11, 'bold')).pack(fill=tk.X, pady=(0, 5))
        self._create_button(btn_frame2, "📊  Compare All 3", self._compare_all,
                           COLORS['orange'], COLORS['bg'], font=('Inter', 10, 'bold')).pack(fill=tk.X)
    
    def _build_center_panel(self):
        """Build the center panel - Feature Inputs."""
        panel = tk.Frame(self.main_frame, bg=COLORS['surface'],
                        highlightbackground=COLORS['border'],
                        highlightthickness=1)
        panel.grid(row=0, column=1, sticky='nsew', padx=5, pady=5)
        
        # Header
        tk.Label(panel,
                text="📝  DONOR FEATURES",
                bg=COLORS['surface'],
                fg=COLORS['accent'],
                font=('IBM Plex Mono', 10, 'bold')).pack(anchor='w', padx=15, pady=(15, 5))
        
        tk.Label(panel,
                text="Set donor features",
                bg=COLORS['surface'],
                fg=COLORS['text'],
                font=('Space Grotesk', 16, 'bold')).pack(anchor='w', padx=15)
        
        tk.Label(panel,
                text="Adjust demographics and financial data",
                bg=COLORS['surface'],
                fg=COLORS['text_secondary'],
                font=('Inter', 9)).pack(anchor='w', padx=15, pady=(0, 10))
        
        # Scrollable frame for features
        canvas = tk.Canvas(panel, bg=COLORS['surface'], highlightthickness=0)
        scrollbar = ttk.Scrollbar(panel, orient='vertical', command=canvas.yview)
        features_frame = tk.Frame(canvas, bg=COLORS['surface'])
        
        features_frame.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox('all')))
        canvas.create_window((0, 0), window=features_frame, anchor='nw', width=380)
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=15, pady=5)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Feature variables
        self.feature_vars: Dict[str, Any] = {}
        
        # Numeric features with sliders
        numeric_features = [
            ('age', 'Age', 17, 90, 1),
            ('education-num', 'Education Num', 1, 16, 1),
            ('capital-gain', 'Capital Gain ($)', 0, 99999, 100),
            ('capital-loss', 'Capital Loss ($)', 0, 4356, 50),
            ('hours-per-week', 'Hours/Week', 1, 99, 1)
        ]
        
        for key, label, min_val, max_val, step in numeric_features:
            self._build_slider(features_frame, key, label, min_val, max_val, step)
        
        # Categorical features
        categorical_features = [
            ('workclass', 'Workclass', WORKCLASS_OPTS),
            ('education_level', 'Education Level', EDUCATION_OPTS),
            ('marital-status', 'Marital Status', MARITAL_OPTS),
            ('occupation', 'Occupation', OCCUPATION_OPTS),
            ('relationship', 'Relationship', RELATIONSHIP_OPTS),
            ('race', 'Race', RACE_OPTS),
            ('sex', 'Sex', SEX_OPTS),
            ('native-country', 'Native Country', COUNTRY_OPTS)
        ]
        
        for key, label, options in categorical_features:
            self._build_dropdown(features_frame, key, label, options)
        
        # Preset buttons
        preset_frame = tk.Frame(panel, bg=COLORS['surface'])
        preset_frame.pack(fill=tk.X, padx=15, pady=10)
        
        self._create_button(preset_frame, "↺ Reset", self._reset_features,
                           COLORS['card'], COLORS['text'], font=('Inter', 9)).pack(side=tk.LEFT, padx=(0, 5))
        self._create_button(preset_frame, "⚄ Random", self._random_features,
                           COLORS['card'], COLORS['text'], font=('Inter', 9)).pack(side=tk.LEFT, padx=(0, 5))
        self._create_button(preset_frame, "★ Typical", self._typical_features,
                           COLORS['purple'], COLORS['white'], font=('Inter', 9)).pack(side=tk.LEFT)
    
    def _build_right_panel(self):
        """Build the right panel - Results & Visualization."""
        panel = tk.Frame(self.main_frame, bg=COLORS['surface'],
                        highlightbackground=COLORS['border'],
                        highlightthickness=1)
        panel.grid(row=0, column=2, sticky='nsew', padx=(10, 0), pady=5)
        
        # Results Section
        tk.Label(panel,
                text="📈  RESULTS",
                bg=COLORS['surface'],
                fg=COLORS['accent'],
                font=('IBM Plex Mono', 10, 'bold')).pack(anchor='w', padx=15, pady=(15, 5))
        
        # Prediction result box
        self.result_box = tk.Frame(panel, bg=COLORS['card'],
                                  highlightbackground=COLORS['border'],
                                  highlightthickness=1)
        self.result_box.pack(fill=tk.X, padx=15, pady=5)
        
        self.result_label = tk.Label(self.result_box,
                                    text="— Run prediction to see result —",
                                    bg=COLORS['card'],
                                    fg=COLORS['text_secondary'],
                                    font=('Space Grotesk', 14, 'bold'))
        self.result_label.pack(pady=20)
        
        # Probability bar
        prob_frame = tk.Frame(panel, bg=COLORS['surface'])
        prob_frame.pack(fill=tk.X, padx=15, pady=5)
        
        tk.Label(prob_frame,
                text="Donor Probability",
                bg=COLORS['surface'],
                fg=COLORS['text_secondary'],
                font=('Inter', 10)).pack(side=tk.LEFT)
        
        self.prob_value = tk.Label(prob_frame,
                                  text="—",
                                  bg=COLORS['surface'],
                                  fg=COLORS['accent'],
                                  font=('IBM Plex Mono', 16, 'bold'))
        self.prob_value.pack(side=tk.RIGHT)
        
        self.prog_bar = ttk.Progressbar(panel, orient='horizontal', mode='determinate', length=200)
        self.prog_bar.pack(fill=tk.X, padx=15, pady=5)
        
        # Predict button
        self._create_button(panel, "🔮  Predict Donor", self._predict,
                           '#4F46E5', COLORS['white'], font=('Inter', 12, 'bold')).pack(fill=tk.X, padx=15, pady=10)
        
        # Separator
        tk.Frame(panel, bg=COLORS['border'], height=1).pack(fill=tk.X, padx=15, pady=10)
        
        # Metrics Section
        tk.Label(panel,
                text="Performance Metrics",
                bg=COLORS['surface'],
                fg=COLORS['text_secondary'],
                font=('Inter', 10, 'bold')).pack(anchor='w', padx=15)
        
        self.metrics_text = ScrolledText(panel,
                                        height=8,
                                        bg=COLORS['card'],
                                        fg=COLORS['text'],
                                        font=('IBM Plex Mono', 9),
                                        relief='flat',
                                        borderwidth=0,
                                        padx=10,
                                        pady=10)
        self.metrics_text.pack(fill=tk.X, padx=15, pady=5)
        self.metrics_text.insert('end', "Train a model to see metrics...\n")
        self.metrics_text.config(state=tk.DISABLED)
        
        # Model scores
        tk.Label(panel,
                text="Model Scores",
                bg=COLORS['surface'],
                fg=COLORS['text_secondary'],
                font=('Inter', 10, 'bold')).pack(anchor='w', padx=15, pady=(10, 5))
        
        scores_frame = tk.Frame(panel, bg=COLORS['surface'])
        scores_frame.pack(fill=tk.X, padx=15, pady=5)
        
        self.score_labels: Dict[str, Tuple[tk.Label, tk.Label]] = {}
        
        for model_name, color in [('Random Forest', COLORS['green']),
                                  ('Logistic Regression', COLORS['purple']),
                                  ('Gradient Boosting', COLORS['orange'])]:
            self._build_score_row(scores_frame, model_name, color)
        
        # Chart (if matplotlib available)
        if MATPLOTLIB_AVAILABLE:
            self.chart_frame = tk.Frame(panel, bg=COLORS['card'], height=200)
            self.chart_frame.pack(fill=tk.X, padx=15, pady=10)
            self.chart_frame.pack_propagate(False)
            
            self.fig = Figure(figsize=(4, 2.5), dpi=80)
            self.fig.patch.set_facecolor(COLORS['card'])
            self.ax = self.fig.add_subplot(111)
            self.ax.set_facecolor(COLORS['card'])
            
            self.canvas = FigureCanvasTkAgg(self.fig, self.chart_frame)
            self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            self._blank_chart()
    
    def _build_status_bar(self):
        """Build the status bar at the bottom."""
        status_frame = tk.Frame(self.root, bg=COLORS['surface'], height=30)
        status_frame.pack(fill=tk.X, side=tk.BOTTOM)
        status_frame.pack_propagate(False)
        
        self.status_label = tk.Label(status_frame,
                                    text="Ready — Load dataset to begin",
                                    bg=COLORS['surface'],
                                    fg=COLORS['text_secondary'],
                                    font=('IBM Plex Mono', 9),
                                    anchor='w')
        self.status_label.pack(side=tk.LEFT, padx=15, pady=5)
    
    # =================================================================
    # UI BUILDER HELPERS
    # =================================================================
    
    def _create_button(self, parent, text, command, bg, fg, font=('Inter', 10)):
        """Create a styled button."""
        btn = tk.Button(parent,
                       text=text,
                       command=command,
                       bg=bg,
                       fg=fg,
                       activebackground=fg,
                       activeforeground=bg,
                       font=font,
                       relief='flat',
                       padx=12,
                       pady=8,
                       cursor='hand2')
        return btn
    
    def _build_model_radio(self, parent, name, color, icon):
        """Build a model selection radio button."""
        frame = tk.Frame(parent, bg=COLORS['card'], padx=10, pady=8)
        frame.pack(fill=tk.X, pady=3)
        
        rb = tk.Radiobutton(frame,
                           text=f"{icon}  {name}",
                           variable=self.selected_model,
                           value=name,
                           bg=COLORS['card'],
                           fg=COLORS['text'],
                           selectcolor=COLORS['surface'],
                           activebackground=COLORS['card'],
                           font=('Inter', 10, 'bold'),
                           anchor='w')
        rb.pack(fill=tk.X)
        
        # Status label for this model
        status_label = tk.Label(frame,
                               text="Not trained",
                               bg=COLORS['card'],
                               fg=COLORS['text_secondary'],
                               font=('IBM Plex Mono', 8))
        status_label.pack(anchor='e')
        
        setattr(self, f'status_{name.replace(" ", "_")}', status_label)
    
    def _build_param_spinbox(self, parent, label, var, min_val, max_val, step):
        """Build a parameter spinbox."""
        frame = tk.Frame(parent, bg=COLORS['surface'])
        frame.pack(fill=tk.X, pady=3)
        
        tk.Label(frame,
                text=label,
                bg=COLORS['surface'],
                fg=COLORS['text_secondary'],
                font=('Inter', 9),
                width=15,
                anchor='w').pack(side=tk.LEFT)
        
        spin = tk.Spinbox(frame,
                         from_=min_val,
                         to=max_val,
                         increment=step,
                         textvariable=var,
                         width=10,
                         bg=COLORS['card'],
                         fg=COLORS['text'],
                         font=('IBM Plex Mono', 9),
                         buttonbackground=COLORS['border'],
                         relief='flat')
        spin.pack(side=tk.LEFT, padx=5)
    
    def _build_slider(self, parent, key, label, min_val, max_val, step):
        """Build a feature slider."""
        frame = tk.Frame(parent, bg=COLORS['surface'])
        frame.pack(fill=tk.X, pady=5)
        
        var = tk.IntVar(value=self.features[key])
        self.feature_vars[key] = var
        
        tk.Label(frame,
                text=label,
                bg=COLORS['surface'],
                fg=COLORS['text_secondary'],
                font=('Inter', 9),
                width=15,
                anchor='w').pack(side=tk.LEFT)
        
        value_label = tk.Label(frame,
                              text=str(var.get()),
                              bg=COLORS['surface'],
                              fg=COLORS['accent'],
                              font=('IBM Plex Mono', 9),
                              width=8)
        value_label.pack(side=tk.RIGHT)
        
        slider = tk.Scale(frame,
                         from_=min_val,
                         to=max_val,
                         resolution=step,
                         orient=tk.HORIZONTAL,
                         bg=COLORS['surface'],
                         fg=COLORS['text'],
                         troughcolor=COLORS['card'],
                         highlightthickness=0,
                         activebackground=COLORS['accent'],
                         sliderrelief='flat',
                         showvalue=False,
                         command=lambda v, k=key, vl=value_label: self._on_slider_change(v, k, vl))
        slider.set(var.get())
        slider.pack(fill=tk.X, padx=5)
    
    def _build_dropdown(self, parent, key, label, options):
        """Build a feature dropdown."""
        frame = tk.Frame(parent, bg=COLORS['surface'])
        frame.pack(fill=tk.X, pady=5)
        
        var = tk.StringVar(value=self.features[key])
        self.feature_vars[key] = var
        
        tk.Label(frame,
                text=label,
                bg=COLORS['surface'],
                fg=COLORS['text_secondary'],
                font=('Inter', 9),
                width=15,
                anchor='w').pack(side=tk.LEFT)
        
        dropdown = ttk.Combobox(frame,
                               textvariable=var,
                               values=options,
                               state='readonly',
                               width=25,
                               font=('Inter', 9))
        dropdown.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        dropdown.bind('<<ComboboxSelected>>', lambda e, k=key: self._on_dropdown_change(k))
    
    def _build_score_row(self, parent, name, color):
        """Build a model score display row."""
        frame = tk.Frame(parent, bg=COLORS['card'], padx=10, pady=5)
        frame.pack(fill=tk.X, pady=2)
        
        name_label = tk.Label(frame,
                             text=name,
                             bg=COLORS['card'],
                             fg=color,
                             font=('Inter', 9, 'bold'))
        name_label.pack(side=tk.LEFT)
        
        f05_label = tk.Label(frame,
                            text="F₀.₅: —",
                            bg=COLORS['card'],
                            fg=COLORS['text_secondary'],
                            font=('IBM Plex Mono', 8))
        f05_label.pack(side=tk.RIGHT)
        
        acc_label = tk.Label(frame,
                            text="Acc: —",
                            bg=COLORS['card'],
                            fg=COLORS['text_secondary'],
                            font=('IBM Plex Mono', 8))
        acc_label.pack(side=tk.RIGHT, padx=10)
        
        self.score_labels[name] = (acc_label, f05_label)
    
    # =================================================================
    # EVENT HANDLERS
    # =================================================================
    
    def _on_slider_change(self, value, key, label):
        """Handle slider value change."""
        val = int(float(value))
        self.features[key] = val
        label.config(text=str(val))
    
    def _on_dropdown_change(self, key):
        """Handle dropdown selection change."""
        self.features[key] = self.feature_vars[key].get()
    
    def _browse_file(self):
        """Open file browser dialog."""
        filepath = filedialog.askopenfilename(
            title="Select census.csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filepath:
            self._load_dataset()
    
    def _load_sample(self):
        """Load sample synthetic dataset."""
        self._load_dataset()
    
    def _load_dataset(self):
        """Load and preprocess the dataset."""
        self._set_status("Generating synthetic dataset...")
        
        # Generate data
        X, y = generate_synthetic_data(32561)
        X_processed, scaler_params = preprocess_features(X)
        
        self.scaler_params = scaler_params
        
        # Split data
        test_size = int(self.model_config['test_size'].get() * len(y))
        split_idx = len(y) - test_size
        
        self.X_train = X_processed[:split_idx]
        self.X_test = X_processed[split_idx:]
        self.y_train = y[:split_idx]
        self.y_test = y[split_idx:]
        
        # Update dataset info
        n_total = len(y)
        n_rich = np.sum(y)
        
        self.dataset_info = {
            'total': n_total,
            'rich': n_rich,
            'train_size': len(self.y_train),
            'test_size': len(self.y_test)
        }
        
        self.dataset_loaded = True
        
        # Update UI
        info_text = (f"{n_total:,} records  |  {n_rich:,} >$50K ({n_rich/n_total:.1%})\n"
                    f"Train: {len(self.y_train):,}  |  Test: {len(self.y_test):,}")
        self.dataset_label.config(text=info_text, fg=COLORS['accent'])
        
        self._set_status(f"Dataset loaded: {n_total:,} records", COLORS['accent'])
    
    def _train_selected(self):
        """Train the selected model."""
        if not self.dataset_loaded:
            messagebox.showwarning("No Dataset", "Please load a dataset first.")
            return
        
        model_name = self.selected_model.get()
        self._train_model(model_name)
    
    def _train_model(self, model_name: str):
        """Train a specific model."""
        if self.is_training:
            return
        
        self.is_training = True
        self._set_status(f"Training {model_name}...")
        
        # Run training in separate thread
        thread = threading.Thread(target=lambda: self._train_worker(model_name))
        thread.daemon = True
        thread.start()
    
    def _train_worker(self, model_name: str):
        """Worker function for model training."""
        try:
            n_est = self.model_config['n_estimators'].get()
            max_d = self.model_config['max_depth'].get()
            lr = self.model_config['learning_rate'].get()
            
            if model_name == 'Random Forest':
                model = SimpleRandomForest(n_estimators=n_est, max_depth=max_d)
            elif model_name == 'Logistic Regression':
                model = SimpleLogisticRegression(max_iter=1000, learning_rate=lr)
            else:  # Gradient Boosting
                model = SimpleGradientBoosting(n_estimators=n_est, max_depth=max_d, learning_rate=lr)
            
            # Train
            model.fit(self.X_train, self.y_train)
            
            # Predict
            y_pred = model.predict(self.X_test)
            
            # Calculate metrics
            metrics = calculate_metrics(self.y_test, y_pred)
            cm = confusion_matrix(self.y_test, y_pred)
            
            # Store results
            self.models[model_name] = model
            self.results[model_name] = {
                'accuracy': metrics['accuracy'],
                'f05': metrics['f05'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'confusion_matrix': cm,
                'predictions': y_pred
            }
            
            # Update UI in main thread
            self.root.after(0, lambda: self._update_training_ui(model_name, metrics))
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Training Error", str(e)))
        finally:
            self.is_training = False
    
    def _update_training_ui(self, model_name: str, metrics: Dict):
        """Update UI after training completes."""
        acc = metrics['accuracy']
        f05 = metrics['f05']
        
        # Update score labels
        acc_label, f05_label = self.score_labels[model_name]
        acc_label.config(text=f"Acc: {acc:.2%}", fg=COLORS['accent'])
        f05_label.config(text=f"F₀.₅: {f05:.4f}", fg=COLORS['accent'])
        
        # Update status label
        status_label = getattr(self, f'status_{model_name.replace(" ", "_")}')
        status_label.config(text=f"✓ Acc: {acc:.2%}", fg=COLORS['green'])
        
        # Update metrics text
        self._update_metrics_text(model_name, metrics)
        
        # Update chart
        if MATPLOTLIB_AVAILABLE:
            self._update_chart(model_name)
        
        self._set_status(f"{model_name} trained — Acc: {acc:.2%}, F₀.₅: {f05:.4f}", COLORS['green'])
    
    def _update_metrics_text(self, model_name: str, metrics: Dict):
        """Update the metrics display."""
        self.metrics_text.config(state=tk.NORMAL)
        self.metrics_text.delete('1.0', tk.END)
        
        cm = self.results[model_name]['confusion_matrix']
        
        text = f"""
{'─' * 40}
  Model: {model_name}
{'─' * 40}
  Accuracy:   {metrics['accuracy']:.4f} ({metrics['accuracy']:.2%})
  F₀.₅ Score: {metrics['f05']:.4f}
  Precision:  {metrics['precision']:.4f}
  Recall:     {metrics['recall']:.4f}
{'─' * 40}
  Confusion Matrix:
              Pred ≤50K   Pred >50K
  Actual ≤50K    {cm[0,0]:5d}      {cm[0,1]:5d}
  Actual >50K    {cm[1,0]:5d}      {cm[1,1]:5d}
{'─' * 40}
"""
        self.metrics_text.insert('end', text)
        self.metrics_text.config(state=tk.DISABLED)
    
    def _compare_all(self):
        """Train and compare all three models."""
        if not self.dataset_loaded:
            messagebox.showwarning("No Dataset", "Please load a dataset first.")
            return
        
        for model_name in ['Random Forest', 'Logistic Regression', 'Gradient Boosting']:
            self._train_model(model_name)
    
    def _predict(self):
        """Make a prediction with current features."""
        model_name = self.selected_model.get()
        model = self.models[model_name]
        
        if model is None:
            messagebox.showwarning("Model Not Trained", f"Please train {model_name} first.")
            return
        
        # Create feature vector
       workclass_val = WORKCLASS_OPTS.index(self.features['workclass']) / len(WORKCLASS_OPTS)
       edu_level_val = EDUCATION_OPTS.index(self.features['education_level']) / len(EDUCATION_OPTS)
       marital_val = MARITAL_OPTS.index(self.features['marital-status']) / len(MARITAL_OPTS)
       sex_val = 1 if self.features['sex'] == 'Female' else 0

        # إنشاء متجه ميزات موسع (أضفنا 4 ميزات جديدة)
       feature_vec = np.array([[
       self.features['age'],
       self.features['education-num'],
       self.features['capital-gain'],
       self.features['capital-loss'],
       self.features['hours-per-week'],
       workclass_val,   # ميزة مضافة
       edu_level_val,   # ميزة مضافة
       marital_val,     # ميزة مضافة
       sex_val          # ميزة مضافة
            ]], dtype=float)
        
        # Preprocess
        feature_vec[:, 2] = np.log1p(feature_vec[:, 2])  # capital-gain
        feature_vec[:, 3] = np.log1p(feature_vec[:, 3])  # capital-loss
        
        # Scale
        if self.scaler_params:
            feature_vec = (feature_vec - self.scaler_params['mins']) / self.scaler_params['ranges']
        
        # Predict
        proba = model.predict_proba(feature_vec)[0]
        label = 1 if proba[1] > 0.5 else 0
        probability = proba[1]
        
        # Update UI
        if label == 1:
            self.result_label.config(
                text="✓  LIKELY DONOR (>50K)",
                fg=COLORS['green']
            )
            self.result_box.config(highlightbackground=COLORS['green'], highlightthickness=2)
        else:
            self.result_label.config(
                text="✗  UNLIKELY DONOR (≤50K)",
                fg=COLORS['red']
            )
            self.result_box.config(highlightbackground=COLORS['red'], highlightthickness=2)
        
        self.prob_value.config(text=f"{probability:.1%}")
        self.prog_bar['value'] = probability * 100
        
        self._set_status(f"Prediction: {'DONOR' if label == 1 else 'NOT DONOR'} ({probability:.1%})",
                        COLORS['green'] if label == 1 else COLORS['red'])
    
    def _reset_features(self):
        """Reset features to defaults."""
        self.features = DEFAULT_FEATURES.copy()
        self._update_feature_ui()
        self._set_status("Features reset to defaults")
    
    def _random_features(self):
        """Generate random features."""
        self.features = {
            'age': random.randint(18, 75),
            'education-num': random.randint(1, 16),
            'capital-gain': 0 if random.random() < 0.8 else random.randint(1000, 50000),
            'capital-loss': 0 if random.random() < 0.9 else random.randint(500, 4000),
            'hours-per-week': random.randint(20, 70),
            'workclass': random.choice(WORKCLASS_OPTS),
            'education_level': random.choice(EDUCATION_OPTS),
            'marital-status': random.choice(MARITAL_OPTS),
            'occupation': random.choice(OCCUPATION_OPTS),
            'relationship': random.choice(RELATIONSHIP_OPTS),
            'race': random.choice(RACE_OPTS),
            'sex': random.choice(SEX_OPTS),
            'native-country': random.choice(COUNTRY_OPTS)
        }
        self._update_feature_ui()
        self._set_status("Random features generated")
    
    def _typical_features(self):
        """Set typical donor features."""
        self.features = TYPICAL_DONOR.copy()
        self._update_feature_ui()
        self._set_status("Typical high-income donor profile loaded")
    
    def _update_feature_ui(self):
        """Update feature UI elements."""
        for key, var in self.feature_vars.items():
            if isinstance(var, tk.IntVar):
                var.set(self.features[key])
            else:
                var.set(self.features[key])
    
    def _set_status(self, message: str, color: str = COLORS['text_secondary']):
        """Update status bar message."""
        self.status_label.config(text=message, fg=color)
    
    def _blank_chart(self):
        """Display blank chart placeholder."""
        self.ax.clear()
        self.ax.set_facecolor(COLORS['card'])
        for spine in self.ax.spines.values():
            spine.set_color(COLORS['border'])
        self.ax.tick_params(colors=COLORS['text_secondary'], labelsize=8)
        self.ax.text(0.5, 0.5, 'Train a model to see chart',
                    ha='center', va='center', color=COLORS['text_secondary'],
                    fontsize=10, transform=self.ax.transAxes)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.canvas.draw()
    
    def _update_chart(self, model_name: str):
        """Update the feature importance chart."""
        model = self.models[model_name]
        if model is None:
            return
        
        self.ax.clear()
        self.ax.set_facecolor(COLORS['card'])
        for spine in self.ax.spines.values():
            spine.set_color(COLORS['border'])
        self.ax.tick_params(colors=COLORS['text_secondary'], labelsize=8)
        
        # Get feature importances or coefficients
        feature_names = ['Age', 'Edu-Num', 'Cap-Gain', 'Cap-Loss', 'Hours']
        
        if model_name == 'Logistic Regression':
            values = np.abs(model.coef_())
            title = "Feature Coefficients (|values|)"
            color = COLORS['purple']
        else:
            values = model.feature_importances(5)
            title = "Feature Importances"
            color = COLORS['green'] if model_name == 'Random Forest' else COLORS['orange']
        
        # Sort by importance
        sorted_idx = np.argsort(values)[::-1]
        sorted_names = [feature_names[i] for i in sorted_idx]
        sorted_values = values[sorted_idx]
        
        bars = self.ax.barh(sorted_names, sorted_values, color=color, alpha=0.8)
        self.ax.set_xlabel('Importance', color=COLORS['text_secondary'], fontsize=9)
        self.ax.set_title(title, color=COLORS['text'], fontsize=10, pad=10)
        
        # Add value labels
        for bar, val in zip(bars, sorted_values):
            self.ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                        f'{val:.3f}', va='center', ha='left',
                        color=COLORS['text_secondary'], fontsize=8)
        
        self.fig.tight_layout()
        self.canvas.draw()


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """Run the CharityML Donor Predictor application."""
    root = tk.Tk()
    
    # Center window on screen
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    x = (screen_width - 1400) // 2
    y = (screen_height - 900) // 2
    root.geometry(f"1400x900+{x}+{y}")
    
    app = CharityMLApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
