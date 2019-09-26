from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

class TimeTransformer(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, X):
        return self

    def transform(self, X):
        new_X = pd.DataFrame()

        for column in X.columns:
            X_radians = self.convert_to_rad( X[column] )
            new_X[column + '_sin'] = np.sin( X_radians )
            new_X[column + '_cos'] = np.cos( X_radians )

        return new_X

    def convert_to_rad(self, c):

        return (np.pi * 2) * (c / np.max(c))

