
import streamlit as st
import seaborn as sns
from joblib import load

@st.cache(allow_output_mutation=True)
def load_model():
    print('loading model')
    return load('./../../sentiment_pipeline.joblib')

model = load_model()

st.title('Live sentiment')

text = st.text_input('what are you thinking?')
# NEG: bad acting, this movie sucks
# POS: wonderful i love it, amazing
sentiment = model.predict_proba([text])
print(sentiment)
st.write(sentiment)

sns.barplot(x=['Negative', 'Positive'], y=sentiment[0])
st.pyplot()