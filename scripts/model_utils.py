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
    confusion_matrix, classification_report
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
import lightgbm as lgbm
import xgboost as xgb

from scripts.config import (
    MODEL_CONFIG, LGBM_PARAMS, RF_PARAMS, XGB_PARAMS,
    HYPERPARAM_GRID, MODELS_DIR, MODEL_FILENAME, PREPROCESSOR_FILENAME
)