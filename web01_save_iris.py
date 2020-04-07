
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from joblib import dump, load

X, y = load_iris(return_X_y=True)

model = DecisionTreeClassifier()

model.fit(X, y)

dump(model, 'iris_model.joblib')
