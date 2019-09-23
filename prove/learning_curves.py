
from sklearn.datasets import load_digits, fetch_olivetti_faces
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from scikitplot.estimators import plot_learning_curve
import matplotlib.pyplot as plt
import numpy as np

# intro teorica
# giro sul playground di google?

dataset = fetch_olivetti_faces()

X = dataset['data']
y = dataset['target']
#np.random.shuffle(y)

#model = MLPClassifier(hidden_layer_sizes=[1], verbose=2) # underfitting
#model = MLPClassifier(hidden_layer_sizes=[100, 50], verbose=2) # fitting
model = MLPClassifier(hidden_layer_sizes=[500, 200], verbose=2) # overfitting

plot_learning_curve(model, X, y)
plt.show()