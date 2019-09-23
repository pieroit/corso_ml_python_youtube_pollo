
# pip install jupyterlab
# 4 tempo trigonometrico

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import QuantileTransformer, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from TimeTransformer import TimeTransformer

df = pd.read_csv('hour.csv')

y = df['cnt']

columns_to_be_deleted = ['cnt', 'casual', 'registered', 'dteday', 'instant']
df.drop(columns_to_be_deleted, axis=1, inplace=True)

transformers = [
    #['one_hot', OneHotEncoder(), ['weathersit']],
    ['scaler', QuantileTransformer(), ['temp', 'atemp', 'hum', 'windspeed']],
    ['time_waver', OneHotEncoder(), ['season', 'yr', 'mnth', 'hr', 'weekday']],
]
ct = ColumnTransformer(transformers, remainder='passthrough')
X = ct.fit_transform(df)

X_train, X_test, y_train, y_test = train_test_split(X, y)

model = MLPRegressor(hidden_layer_sizes=[50, 10], verbose=True)
model.fit(X_train, y_train)

p_train = model.predict(X_train)
p_test  = model.predict(X_test)
mae_train = mean_absolute_error(y_train, p_train)
mae_test  = mean_absolute_error(y_test, p_test)

print( f'mean y {np.mean(y)}' )
print( f'Train {mae_train}, test {mae_test}' )


