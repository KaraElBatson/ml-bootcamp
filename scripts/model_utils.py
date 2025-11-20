"""
Model training and evaluation utilities
"""
import os
import time
import joblib
import warnings
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve, precision_recall_curve,
    confusion_matrix, classification_report, mean_absolute_error,
    mean_squared_error, r2_score
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
import lightgbm as lgbm
import xgboost as xgb

from sklearn.svm import SVR

from scripts.config import (
    MODEL_CONFIG, LGBM_PARAMS, RF_PARAMS, XGB_PARAMS, SVR_PARAMS,
    HYPERPARAM_GRID, MODELS_DIR, MODEL_FILENAME, PREPROCESSOR_FILENAME
)

def evaluate_regression_model(model, X_train, X_test, y_train, y_test, preprocessor=None, model_name="Model"):
    """Evaluate a regression model and return metrics"""
    if preprocessor:
        X_train = preprocessor.fit_transform(X_train)
        X_test = preprocessor.transform(X_test)
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    return {
        "model_name": model_name,
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2
    }
