
# DA FARE IN UN NOTEBOOK?

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor

x1 = np.arange(0, 10, 0.5)
n = len(x1)
X = np.expand_dims(x1, axis=1)
y = np.cos(x1) + (np.random.random(n))

#model = MLPRegressor(hidden_layer_sizes=[1], max_iter=10000, tol=-1, verbose=2) # underfitting
#model = MLPRegressor(hidden_layer_sizes=[10], max_iter=10000, tol=-1, verbose=2) # fitting
model = MLPRegressor(hidden_layer_sizes=[100, 100, 100, 100], max_iter=10000, tol=-1, verbose=2) # overfitting
model.fit(X, y)
p = model.predict(X)

sns.scatterplot(x=x1, y=y)
sns.lineplot(x=x1, y=p)
plt.show()
