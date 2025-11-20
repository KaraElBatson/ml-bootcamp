"""
Data loading and basic preprocessing utilities
"""
from typing import Tuple, List
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from scripts.config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    MODEL_CONFIG,
)


def load_csv(file_name: str) -> pd.DataFrame:
    """Load a CSV file from data/raw by file name"""
    file_path = os.path.join(RAW_DATA_DIR, file_name)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    return pd.read_csv(file_path)


def split_data(df: pd.DataFrame, target: str = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split data into train and test sets"""
    target = target or MODEL_CONFIG["target_column"]
    X = df.drop(columns=[target])
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=MODEL_CONFIG["test_size"],
        random_state=MODEL_CONFIG["random_state"],
        stratify=y if y.nunique() <= 10 else None
    )
    return X_train, X_test, y_train, y_test


def build_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    """Build a preprocessing pipeline for numeric and categorical features"""
    numeric_features = df.select_dtypes(include=["int", "float"]).columns.tolist()
    categorical_features = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )
    return preprocessor
