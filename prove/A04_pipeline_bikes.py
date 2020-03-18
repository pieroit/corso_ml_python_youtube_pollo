


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import OneHotEncoder
from joblib import dump

df = pd.read_csv('hour.csv')

categorical_feats = ['weathersit','season', 'yr', 'mnth', 'hr', 'weekday']
numerical_feats   = ['temp', 'atemp', 'hum', 'windspeed']
X = df[ categorical_feats + numerical_feats ]
y = df['cnt']

X_train, X_test, y_train, y_test = train_test_split(X, y)

transformers = [
    ['one_hot', OneHotEncoder(), categorical_feats],
    ['scaler', RobustScaler(), numerical_feats]
]
ct = ColumnTransformer(transformers)

steps = [
    ['column_trans', ct],
    #['model', LinearRegression()]
    ['model', MLPRegressor(hidden_layer_sizes=[50, 50], verbose=2)]
]
pipeline = Pipeline(steps)

# SAY: differenza tra ColumnTransformer e Pipeline
# SAY: puoi mettere come feature texsto, numeri e categorie contemporaneamente

pipeline.fit(X_train, y_train)

p_train = pipeline.predict(X_train)
p_test  = pipeline.predict(X_test)
mae_train = mean_absolute_error(y_train, p_train)
mae_test  = mean_absolute_error(y_test, p_test)

print(f'Median cnt {np.median(y)}')
print( f'Train {mae_train}, test {mae_test}' )

dump(pipeline, 'bike_pipeline.joblib')