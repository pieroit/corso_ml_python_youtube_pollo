# download from https://archive.ics.uci.edu/ml/machine-learning-databases/00275/

# 1 feature non vettorizzate e non scalate
# 2 vettorizza categorie
# 3 scala var numeriche

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import QuantileTransformer, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression

df = pd.read_csv('hour.csv')

y = df['cnt']

columns_to_be_deleted = ['cnt', 'casual', 'registered', 'dteday', 'instant']
df.drop(columns_to_be_deleted, axis=1, inplace=True)
print(y)
print(df.columns)

transformers = [
    ['one_hot', OneHotEncoder(), ['season', 'yr', 'mnth', 'hr', 'weekday', 'weathersit']],
    ['scaler', QuantileTransformer(), ['temp', 'atemp', 'hum', 'windspeed']]
]
ct = ColumnTransformer(transformers, remainder='passthrough')
X = ct.fit_transform(df)

#print(df.iloc(45))
#print()

X_train, X_test, y_train, y_test = train_test_split(X, y)

model = LinearRegression()
model.fit(X_train, y_train)

p_train = model.predict(X_train)
p_test  = model.predict(X_test)
mae_train = mean_absolute_error(y_train, p_train)
mae_test  = mean_absolute_error(y_test, p_test)

print( f'mean y {np.mean(y)}' )
print( f'Train {mae_train}, test {mae_test}' )


