import streamlit as st
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import pickle

st.write("""
# Ekobilet Machine Learning
beta -v1

Prognoza **ceny biletu** wg: miasta, typu i dnia wydarzania 
""")
st.write('---')


# Sidebar
# Header of Specify Input Parameters
st.sidebar.header('Określ parametry wejściowe')


def user_input_features():
    typ = st.sidebar.selectbox(
        "Wybierz typ wydarzenia?",
        ("Koncert", "Kino")
    )
    dzien = st.sidebar.selectbox(
        "Wybierz dzień wydarzenia",
        ("W tygodniu", "Weekend")
    )
    lokalizacja = st.sidebar.selectbox(
        "Wybierz miasto",
        ("Kraków", "Stary Sącz")
    )

    data = {"Typ":typ,
            "Dzień": dzien,
            "Lokalizacja": lokalizacja,
            }
    features = pd.DataFrame(data, index=[0])
    return features



# Main Panel

@st.cache
def load_data_row():
    data = pd.read_pickle('df_row.pickle')
    return data

df = load_data_row


# prediction data
df_prediction = user_input_features()

st.header('Wybrane parametry do prognozy')
st.write(df_prediction)
st.write('---')


df_prediction["Typ"] = df_prediction["Typ"].replace({
    "Kino":0,
    "Koncert":1,
})

df_prediction["Dzień"] = df_prediction["Dzień"].replace({
    "W tygodniu":0,
    "Weekend":1,
})

df_prediction["Lokalizacja"] = df_prediction["Lokalizacja"].replace({
    "Stary Sącz":0,
    "Kraków":1,
})
#
# st.header('Prediction of dataset')
# st.write(df_prediction)

# Apply Model to Make Prediction
with open('eko_model.pickle','rb') as f:
  model = pickle.load(f)

prediction = model.predict(df_prediction)

st.header('Prognoza ceny biletu')
st.write(f"{round(prediction[0],2)} zł")
st.write('---')

# df_row = load_data_row
# st.header('Dane użyte treningu modelu')
# st.write(df_row)
# st.write('---')
