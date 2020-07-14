
import streamlit as st
import pandas as pd
import seaborn as sns


@st.cache
def load_data():
    print('load data')
    df = pd.read_csv('./../hour.csv')
    return df

st.title('Load pandas dataframe')


data = load_data()
st.write(data)

weekday = st.number_input('Choose weekday', value=0, min_value=0, max_value=6)
st.write('Weekday', weekday)
wd_filter = data['weekday'] == weekday
st.write( data[ wd_filter ] )
st.write( data[ wd_filter ].describe() )

sns.barplot(data=data[ wd_filter ], x='hr', y='cnt')
st.pyplot()