"""
Configuration file for the Machine Learning Bootcamp Project
"""

import os
import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

# Data paths
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
EXTERNAL_DATA_DIR = os.path.join(DATA_DIR, "external")

# Directories for notebook outputs
NOTEBOOKS_DIR = os.path.join(PROJECT_ROOT, "notebooks")
ASSETS_DIR = os.path.join(PROJECT_ROOT, "assets")
MODELS_DIR = os.path.join(ASSETS_DIR, "models")
IMAGES_DIR = os.path.join(ASSETS_DIR, "images")

# Model configuration
MODEL_CONFIG = {
    "target_column": "credit_score",
    "test_size": 0.25,
    "random_state": 42,
    "cross_validation_folds": 5,
    "scoring_metric": "roc_auc"  # For binary classification
}

# Data configuration
DATA_CONFIG = {
    "missing_threshold": 0.5,  # Drop columns with >50% missing values
    "correlation_threshold": 0.95,  # Remove one of columns with correlation >0.95
    "outlier_method": "iqr",  # Method for outlier detection
    "outlier_threshold": 1.5  # IQR outlier threshold
}

# Model parameters
LGBM_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "max_depth": -1,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1
}

RF_PARAMS = {
    "n_estimators": 100,
    "max_depth": 10,
    "random_state": 42,
    "n_jobs": -1,
    "class_weight": "balanced"
}

XGB_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "n_estimators": 100,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "n_jobs": -1
}

# Hyperparameter tuning grid
HYPERPARAM_GRID = {
    "lgbm": {
        "num_leaves": [31, 50, 100],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [-1, 10, 20],
        "n_estimators": [100, 200, 500]
    },
    "rf": {
        "n_estimators": [100, 200, 500],
        "max_depth": [5, 10, 20, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4]
    },
    "xgb": {
        "n_estimators": [100, 200, 500],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample": [0.8, 0.9, 1.0],
        "colsample_bytree": [0.8, 0.9, 1.0]
    }
}

# Feature engineering settings
FEATURE_CONFIG = {
    "scale_numeric": True,
    "one_hot_encode_categorical": True,
    "pca_components": None,  # Set to number for dimensionality reduction
    "feature_importance_threshold": 0.01  # Minimum importance for feature selection
}

# Model persistence
MODEL_FILENAME = "credit_scoring_model.pkl"
PREPROCESSOR_FILENAME = "preprocessor.pkl"

# Logging configuration
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE = os.path.join(PROJECT_ROOT, "ml_bootcamp.log")

# API Configuration
API_CONFIG = {
    "host": "0.0.0.0",
    "port": 8000,
    "debug": False
}