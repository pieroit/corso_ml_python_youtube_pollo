
import numpy as np
np.random.seed(0)
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error

X = np.arange(0, 10, 0.2)
n = len(X)
y = np.cos(X) + (2 * np.random.random(n))
X = np.expand_dims(X, axis=1)


X_train, X_test, y_train, y_test = train_test_split(X, y)

#model = MLPRegressor(hidden_layer_sizes=[1], max_iter=10000, tol=-1, verbose=2) # underfitting
model = MLPRegressor(hidden_layer_sizes=[50], max_iter=10000, tol=-1, verbose=2) # fitting
#model = MLPRegressor(hidden_layer_sizes=[100, 100, 100, 100], max_iter=10000, tol=-1, verbose=2) # overfitting
model.fit(X_train, y_train)

p_train = model.predict(X_train)
p_test  = model.predict(X_test)
p       = model.predict(X)  # ci serviranno per il grafico

mae_train = mean_absolute_error(y_train, p_train)
mae_test  = mean_absolute_error(y_test, p_test)
print( f'Train {mae_train}, Test {mae_test}' )


sns.scatterplot(x=X_train[:,0], y=y_train)
sns.scatterplot(x=X_test[:,0], y=y_test)
sns.lineplot(x=X[:,0], y=p)
plt.show()
