

import streamlit as st
import pandas as pd
import seaborn as sns

@st.cache
def load_data():
    print('loading data')
    df = pd.read_csv('./../hour.csv')
    return df

st.title('Load pandas DataFrame')

data = load_data()
st.write(data.head())

weekday = st.number_input(
    'Choose weekday',
    value=0,
    min_value=0,
    max_value=6
)

st.write(weekday)

data_filter = data['weekday'] == weekday
filtered_data = data[ data_filter ]

st.write( filtered_data )
st.write( filtered_data.describe() )

sns.barplot(data=filtered_data, x='hr', y='cnt')
st.pyplot()
















































