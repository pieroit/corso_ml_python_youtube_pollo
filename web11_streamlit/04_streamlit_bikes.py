
import streamlit as st
from joblib import load
import pandas as pd
import plotly.graph_objects as go

@st.cache(allow_output_mutation=True)
def load_model():
    print('load model')
    return load('./../bikes_pipeline.joblib')

@st.cache
def load_data():
    print('load data')
    return pd.read_csv('./../hour.csv')


model = load_model()
data = load_data()

st.write(model)

st.title('Bike booking prediction')

user_input = {}

categoricals = ['weathersit', 'season', 'yr', 'mnth', 'hr', 'weekday']
for feat in categoricals:
    unique_values = data[feat].unique()
    user_input[feat] = st.sidebar.selectbox(feat, unique_values)

numericals = ['temp', 'atemp', 'hum', 'windspeed']
for feat in numericals:
    v_min = float( data[feat].min() )
    v_max = float( data[feat].max() )
    user_input[feat] = st.sidebar.slider(
        feat,
        min_value=v_min,
        max_value=v_max,
        value= (v_min + v_max) / 2
    )

X = pd.DataFrame([user_input])
st.write(X)

prediction = model.predict(X)
st.write(prediction)


fig = go.Figure(
    go.Indicator(
        mode = 'gauge+number',
        value = prediction[0],
        gauge = {
            #'shape': 'bullet',
            'axis': { 'range': [0, 500]}
        }
    ),
)

st.plotly_chart(fig)



































































































