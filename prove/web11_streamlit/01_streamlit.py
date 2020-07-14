
'''
## Streamlit

- permette di costruire esperienze interattive in python
 (incluso server web e frontend)
- lo script viene rieseguito a daccapo ogni volta che qualcosa cambia nell'interfaccia
- widget si utilizzano come variabili
- cache per non dover ricaricare ogni volta i dati
'''


# pip install streamlit pandas scikit-learn
# streamlit hello
# cd <cartella progetto >
# streamlit run web11_streamlit.py

import streamlit as st

# scrivere a schermo
st.title('Streamlit demo')
st.write('let\'s try it')

# variabili
x = 4
st.write( x, 'cube is', x*x*x ) # st.write manda le cose a schermo

# input
y = st.slider('y')
st.write('You just chose y =', y)

# input sidebar
z = st.sidebar.slider('z')
st.write('z from the sidebar is', z)

st.write('y + z =', y+z)

city = st.selectbox(
    'Scegli una città',
    ['Roma', 'Milano', 'Napoli']
)

st.write('You want to travel to', city)
