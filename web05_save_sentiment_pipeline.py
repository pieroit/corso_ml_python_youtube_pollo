
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score
from joblib import dump
from sklearn.pipeline import Pipeline

df = pd.read_csv('movie_review.csv')
print(df.head())

X = df['text']
y = df['tag']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4)

steps = [
    ('vectorizer', CountVectorizer()),
    ('model', BernoulliNB())
]
pipeline = Pipeline(steps)

pipeline.fit(X_train, y_train)

p_train = pipeline.predict(X_train)
p_test = pipeline.predict(X_test)

acc_train = accuracy_score(y_train, p_train)
acc_test = accuracy_score(y_test, p_test)

print( f'Train acc. {acc_train}, test acc. {acc_test}' )

#dump(vect, 'sentiment_vectorizer.joblib')
#dump(model, 'sentiment_model.joblib')

dump(pipeline, 'sentiment_pipeline.joblib')
