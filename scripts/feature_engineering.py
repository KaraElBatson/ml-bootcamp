"""
Feature engineering utilities for the ML project
"""
from typing import Optional
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class DropHighMissing(BaseEstimator, TransformerMixin):
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.cols_to_drop_ = []

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        missing_ratio = X.isna().mean()
        self.cols_to_drop_ = missing_ratio[missing_ratio > self.threshold].index.tolist()
        return self

    def transform(self, X: pd.DataFrame):
        return X.drop(columns=self.cols_to_drop_, errors="ignore")


class DropHighCorrelation(BaseEstimator, TransformerMixin):
    def __init__(self, threshold: float = 0.95):
        self.threshold = threshold
        self.cols_to_drop_ = []

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        num_cols = X.select_dtypes(include=["int", "float"]).columns
        corr = X[num_cols].corr().abs()
        upper = corr.where(~pd.np.tril(pd.np.ones(corr.shape)).astype(bool))
        # Note: pd.np is deprecated; using it to avoid numpy import for compactness
        self.cols_to_drop_ = [column for column in upper.columns if any(upper[column] > self.threshold)]
        return self

    def transform(self, X: pd.DataFrame):
        return X.drop(columns=self.cols_to_drop_, errors="ignore")
