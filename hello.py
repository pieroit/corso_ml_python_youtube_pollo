
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import numpy as np
np.random.seed(2)

dataset = load_boston()

X = dataset['data']
y = dataset['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

model = LinearRegression()
model.fit(X_train, y_train)

p_train = model.predict(X_train)
p_test = model.predict(X_test)

mae_train = mean_absolute_error(y_train, p_train)
print('MAE train', mae_train)
mae_test = mean_absolute_error(y_test, p_test)
print('MAE test', mae_test)



