import streamlit as st
from sktime.base import BaseEstimator

@st.cache_resource
def load_model(model_file: str) -> BaseEstimator:
    filepath = 'models/' + model_file + '.zip'
    model = BaseEstimator().load_from_path(filepath)
    return model
