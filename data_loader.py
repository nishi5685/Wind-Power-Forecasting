import pandas as pd
import streamlit as st

@st.cache_data
def load_data() -> pd.DataFrame:
    # Load dataframe from pickle
    df = pd.read_pickle('data/df.pkl')
    df = df['2021-01-01 00:00:00':]
    return df
