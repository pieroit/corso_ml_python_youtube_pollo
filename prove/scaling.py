
# terza forma principale di data prep, oltre a vettorizzazione e valori mancanti

# intro leggera: la scalatura è una buona pratica per far decidere all'algoritmo di addestramento
# quali feature sono importanti senza risentire della scala numerica e della distribuzione

# vediamo la cosa visivamente: prendiamo 2 feature
# gli algoritmi (knn, gradient descent) che si basano sulla distanza tra i puntini nello spazio risentono delle scale
# dobbiamo fare in modo che tutti i puntini siano in una nuvola con assi di range simile

from sklearn.datasets import load_wine
from sklearn.preprocessing import MinMaxScaler, StandardScaler, QuantileTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

dataset = load_wine()
print(dataset['DESCR'])

#print(dataset['data'])
X = dataset['data'][:, [4, 7]]
#print(X)

df = pd.DataFrame(X, columns=['magnesium', 'phenols'])
#g = sns.scatterplot(data=df, x='magnesium', y='phenols')
g = sns.pairplot(df)
#g.set( xlim=(-10, 200), ylim=(-10, 200))
plt.show()

scaler = MinMaxScaler(feature_range=(-1,1))
#scaler = StandardScaler()
#scaler = QuantileTransformer()
X_scaled  = scaler.fit_transform(df)
df_scaled = pd.DataFrame(X_scaled, columns=['magnesium', 'phenols'])
sns.pairplot(df_scaled)
plt.show()


# ESEMPIO PRATICO con kNN

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

