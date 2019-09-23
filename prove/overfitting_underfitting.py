
from sklearn.neural_network import MLPClassifier
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# fai vedere slide
# overfit = imparo a memoria
# underfit = non imparo un cazzo
# in questo script non c'è fitting perchè i dati sono a caso!

n = 100
X = np.random.random(size=(n, 5))
y = np.random.choice(['sì', 'no'], size=n)

X_train, X_test, y_train, y_test = train_test_split(X, y)

model = MLPClassifier(hidden_layer_sizes=[1000, 500], verbose=2)
#model = MLPClassifier(hidden_layer_sizes=[1], verbose=2) # low resource
model.fit(X_train, y_train)

p_train = model.predict(X_train)
p_test  = model.predict(X_test)

acc_train = accuracy_score(y_train, p_train)
acc_test  = accuracy_score(y_test, p_test)

print(f'train {acc_train}, test {acc_test}')

