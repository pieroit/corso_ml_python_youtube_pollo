from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

class TimeTransformer(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def convert_to_rad(self, s):
        return (np.pi * 2) * (s/np.max(s))

    def fit(self, X):
        return self

    def transform(self, X):

        new_X = pd.DataFrame()

        for c in X.columns:
            X_radians = self.convert_to_rad( X[c] )
            new_X[ c + '_sin' ] = np.sin(X_radians)
            new_X[ c + '_cos' ] = np.cos(X_radians)

        return new_X