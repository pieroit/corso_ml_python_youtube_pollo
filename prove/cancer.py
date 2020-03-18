
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

dataset = load_breast_cancer()
X = dataset['data']
y = dataset['target']

X_train, X_test, y_train, y_test = train_test_split(X, y)

model = LogisticRegression()
model.fit(X_train, y_train)

p_train = model.predict(X_train)
p_test  = model.predict(X_test)

acc_train = accuracy_score(y_train, p_train)
acc_test  = accuracy_score(y_test, p_test)

print( f'Train accuracy {acc_train}, test accuracy {acc_test}' )