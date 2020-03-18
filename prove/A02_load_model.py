
from joblib import load

model = load('model.joblib')

# nuovi dati

# ERRORE(vanno impostati come matrice)
#text = 'This movie sucks I hate it. It was a nightmare seeing it.'
#model.predict(text)

texts = ['This movie sucks I hate it. It was a nightmare seeing it.']
# ERRORE Dobbiamo ripreparare i dati
#model.predict(texts)

vect = load('vectorizer.joblib')
X_new = vect.transform(texts)    # don't do fit !!!
p = model.predict(X_new)
prob = model.predict_proba(X_new)

print(p, prob)
# aggiungere anche una frase positiva

# ispezione da notebook