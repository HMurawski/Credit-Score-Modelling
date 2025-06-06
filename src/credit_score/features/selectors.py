"""
Transformers that require .fit() on the training dataset.
"""

from __future__ import annotations  # Enables modern typing: list[str] in Python 3.8+
import pandas as pd
from typing import List, Sequence
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from statsmodels.stats.outliers_influence import variance_inflation_factor


class VIFDropper(BaseEstimator, TransformerMixin):
    """
    Iteratively drops features with VIF (Variance Inflation Factor) above a given threshold.

    Parameters
    ----------
    threshold : float, default=5.0
        Maximum acceptable VIF. Columns with higher values will be dropped.
    
    Attributes
    ----------
    keep_ : list[str]
        List of retained columns after applying VIF filtering.
    """
    def __init__(self, threshold: float = 5.0):
        self.threshold = threshold
        self.keep_: List[str] = []

    def fit(self, X, y=None):
        df = pd.DataFrame(X)
        # Ensure only numerical features are evaluated
        df = df.select_dtypes(include="number").replace([np.inf, -np.inf], np.nan).dropna() 
        if df.shape[1] == 0:
            self.keep_ = []
            return self
        cols: List[str] = df.columns.tolist()

        while True and cols:  # safeguard in case all columns are dropped
            vifs = [
                variance_inflation_factor(df[cols].values, i)
                for i in range(len(cols))
            ]
            max_vif = max(vifs)
            if max_vif < self.threshold:
                break
            drop = cols.pop(vifs.index(max_vif))

        self.keep_ = cols
        return self

    def transform(self, X):
        # Will raise KeyError if some expected columns are missing — intended behavior
        return pd.DataFrame(X)[self.keep_]


# --------------------------------------------------------------------- #
class ColumnSubsetter(BaseEstimator, TransformerMixin):
    """
    Selects and passes through only a predefined subset of columns.

    Parameters
    ----------
    cols : Sequence[str]
        List of column names to retain.
    """
    def __init__(self, cols: Sequence[str]):
        self.cols = list(cols)

    def fit(self, X, y=None):
        # No fitting required
        return self

    def transform(self, X):
        return pd.DataFrame(X)[self.cols].copy()
