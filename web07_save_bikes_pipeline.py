

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from joblib import dump
from sklearn.pipeline import Pipeline

df = pd.read_csv('hour.csv')
y = df['cnt']

categorical_features = ['weathersit', 'season', 'yr', 'mnth', 'hr', 'weekday']
numerical_features = ['temp', 'atemp', 'hum', 'windspeed']

X = df[ categorical_features + numerical_features ]
y = df['cnt']

X_train, X_test, y_train, y_test = train_test_split(X, y)

transformers = [
    ['one_hot', OneHotEncoder(), categorical_features],
    ['scaler', RobustScaler(), numerical_features]
]
ct = ColumnTransformer(transformers)

steps = [
    ['column_transformer', ct],
    ['model', LinearRegression()]
]
pipeline = Pipeline(steps)

pipeline.fit(X_train, y_train)

p_train = pipeline.predict(X_train)
p_test  = pipeline.predict(X_test)
mae_train = mean_absolute_error(y_train, p_train)
mae_test  = mean_absolute_error(y_test, p_test)

print(f'Median cnt {np.median(y)}')
print( f'Train {mae_train}, test {mae_test}' )

dump(pipeline, 'bikes_pipeline.joblib')
