
from sklearn.feature_extraction.text import CountVectorizer

X = [
    'ciao ciao miao',
    'miao',
    'miao bao'
]

vectorizer = CountVectorizer()
vectorizer.fit(X)
X = vectorizer.transform(X)

print( vectorizer.get_feature_names() )
print(X.todense())
print(X)

