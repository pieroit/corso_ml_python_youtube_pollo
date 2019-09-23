
# perchè scaler (distanze e variabilità omogenee)
# vari tipi di scaler
# grafico finale
# ESERCIZIO: prova di convergenza più veloce?

from sklearn.datasets import load_wine
from sklearn.preprocessing import MinMaxScaler, StandardScaler, QuantileTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from time import time

dataset = load_wine()
print(dataset['DESCR'])

#print(dataset['data'])
X = dataset['data'][:, [4, 10]]
print(X)

df = pd.DataFrame(X, columns=['mag', 'ghj'])
sns.pairplot(df)
#plt.show()

#scaler = MinMaxScaler(feature_range=(-1,1))
#scaler = StandardScaler()
scaler = QuantileTransformer()
X_scaled  = scaler.fit_transform(df)
df_scaled = pd.DataFrame(X_scaled, columns=['mag', 'ghj'])
sns.pairplot(df_scaled)
plt.show()

X = dataset['data']
y = dataset['target']
model = KNeighborsClassifier()
model.fit(X, y)
p = model.predict(X)
print( f'Accuracy (not scaled): { accuracy_score(y, p) }' )


X = dataset['data']
y = dataset['target']
scaler = StandardScaler()
X = scaler.fit_transform(X)
model = KNeighborsClassifier()
model.fit(X, y)
p = model.predict(X)
print( f'Accuracy (scaled): { accuracy_score(y, p) }' )

