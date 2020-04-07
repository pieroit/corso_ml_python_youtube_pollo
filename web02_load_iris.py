
from joblib import load

model = load('iris_model.joblib')

X = [
    [2.0, 3.0, 1.4, 5.2]
]

p = model.predict(X)

print(p)