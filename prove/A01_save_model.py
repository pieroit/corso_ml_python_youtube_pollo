

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score

# importa libreria per salvare il modello
from joblib import dump

df = pd.read_csv('movie_review.csv')
print(df.head())

X = df['text']
y = df['tag']

X_train, X_test, y_train, y_test = train_test_split(X, y)

# cominciamo a fare le cose a modino fittando il vettorizzatore solo sul training
vect = CountVectorizer()
vect.fit(X_train)
X_train = vect.transform(X_train)
X_test  = vect.transform(X_test)

model = BernoulliNB()
model.fit(X_train, y_train)

p_train = model.predict(X_train)
p_test = model.predict(X_test)

acc_train = accuracy_score(y_train, p_train)
acc_test = accuracy_score(y_test, p_test)

print( f'Train acc. {acc_train}, test acc. {acc_test}' )

dump(model, 'model.joblib')

# da aggiungere nel terzo video
dump(vect, 'vectorizer.joblib')
