
import streamlit as st

st.title('Streamlit demo')

st.write('ciao tutti polletti!!!!')

x = 4
st.write('x cube is ', x * x * x)

y = st.slider('slider per y')
st.write('selected value', y)

z = st.sidebar.slider('altro slider')

st.write('x + y + z', x + y + z)

city = st.selectbox(
    'scegli una città',
    ['Roma', 'Milano', 'Napoli']
)

st.write('Hai selezionato', city)































