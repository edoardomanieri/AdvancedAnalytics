import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.base import TransformerMixin, BaseEstimator

class CustomScaler(BaseEstimator, TransformerMixin):
    def __init__(self, num_cols=None):
        self.num_cols = num_cols
        self.scaler = StandardScaler()

    def fit(self, X, y=None):
        self.scaler.fit(X[:,self.num_cols])
        return self

    def transform(self, X):
        a = X.copy()
        a[:,self.num_cols] = self.scaler.transform(a[:,self.num_cols])
        return a
