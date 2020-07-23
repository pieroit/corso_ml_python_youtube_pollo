
import streamlit as st
import seaborn as sns
from joblib import load


@st.cache(allow_output_mutation=True)
def load_model():
    print('loading model')
    pipeline = load('./../sentiment_pipeline.joblib')
    return pipeline


model = load_model()

st.title('Live sentiment')

text = st.text_input('What do you think about this movie?')

sentiment = model.predict_proba([text])
st.write(sentiment)

sns.barplot(x=['Negative', 'Positive'], y=sentiment[0])
st.pyplot()








































