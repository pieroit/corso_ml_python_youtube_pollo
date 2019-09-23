from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

X = [
    [110, 1.70, 'rugby'],
    [100, 1.90, 'basket'],
    [120, 1.90, 'rugby'],
    [ 70, 1.60, 'soccer']
]

transformers = [
    ['cvxz', OneHotEncoder(), [2]],
]
ct = ColumnTransformer(transformers, remainder='passthrough')

ct.fit(X)
X = ct.transform(X)

print(X)