from sklearn.datasets import load_boston #funzione
from sklearn.linear_model import LinearRegression #classe
from  sklearn.metrics import mean_absolute_error #misura di errore sempre positiva, piu è vicina allo 0 meglio è
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from mytrainers import  MyLinearRegression

import numpy as np
np.random.seed(12)
from joblib import dump, load

dataset = load_boston()

#print(dataset['DESCR'])
#print(dataset['data'])
#print(dataset['target'])

X = dataset['data']
y = dataset['target']

# separate data ub train and test - validation -
X_train, X_test, y_train, y_test = train_test_split(X, y)


model = MyLinearRegression()  #DecisionTreeRegressor() #LinearRegression() # sarebbe la retta

model.fit(X_train, y_train)
#model.fit(X, y) # raddrizzamento retta in base ai punti

p_train = model.predict(X_train)
p_test = model.predict(X_test)   #p = model.predict(X)
mae_train = mean_absolute_error(y_train, p_train)
mae_test = mean_absolute_error(y_test, p_test)

#mae = mean_absolute_error(y, p)

print(f'Train Error: {mae_train}')
print(f'Test Error: {mae_test}') # f è un interpolazione di stringa


#dump(model, 'boston_model.joblib')


