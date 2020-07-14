
import streamlit as st
from joblib import load
import pandas as pd
#import seaborn as sns
#import matplotlib.pyplot as plt

# pip install plotly
import plotly.graph_objects as go

@st.cache(allow_output_mutation=True)
def load_model():
    print('loading model')
    return load('./../../bikes_pipeline.joblib')

@st.cache()
def load_data():
    print('loading data')
    return pd.read_csv('./../hour.csv')

model = load_model()
data  = load_data()

# far vedere step by step
st.write(model['column_transformer'])

st.title('Bike booking prediction')

user_input = {}

categoricals = ['weathersit', 'season', 'yr', 'mnth', 'hr', 'weekday']
for feat in categoricals:
    values = data[feat].unique()
    user_input[feat] = st.sidebar.selectbox(feat, values)

numericals = ['temp', 'atemp', 'hum', 'windspeed']
for feat in numericals:
    min = float( data[feat].min() )
    max = float( data[feat].max() )
    user_input[feat] = st.sidebar.slider(feat, min_value=min, max_value=max)

print(user_input)
X = pd.DataFrame([user_input])
st.write('User input', X)

prediction = model.predict(X)
print(prediction)

# SEABORN
#st.markdown(f'# { round(prediction[0]) } bookings')
#sns.barplot(x=prediction, y=['bookings'], orient='horizontal')
#plt.xlim(0, 500)
#st.pyplot()

# PLOTLY
# https://plotly.com/python/bullet-charts/
fig = go.Figure(go.Indicator(
    mode = "gauge+number",
    gauge = {
        #'shape': 'bullet',
        'axis' : { 'range': [0, 500] }
    },
    value  = prediction[0]
))
st.plotly_chart(fig)