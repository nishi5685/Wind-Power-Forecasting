import streamlit as st
import plotly.express as px
import numpy as np
import pandas as pd


from datetime import datetime
from utils.model_dict import model_dict
from utils.data_loader import load_data
from utils.model_loader import load_model
from utils.forecaster import get_forecast
from utils.metrics_calc import get_metrics

st.set_page_config(page_title = "Wind Power Forecasting", page_icon="🌀", layout="wide")

st.title('Wind Power Forecasting 🌀')
st.markdown(
    '''
    Wind power forecasting using machine learning for day-ahead energy trading.  
    Dataset: [Kelmarsh Wind Farm Data](https://zenodo.org/record/7212475)
    '''
)

# Sidebar controls
st.sidebar.markdown('')
st.sidebar.header('Controls')
model_name = st.sidebar.radio(
    'Select model', 
    options=model_dict.keys(), 
    index=7)
fd = st.sidebar.slider(
    'Forecast date', 
    min_value=datetime(2021, 4, 1), 
    max_value=datetime(2021, 6, 30), 
    value=datetime(2021, 5, 1))
fh = st.sidebar.slider(
    'Forecast horizon (hours)', 
    min_value=1, 
    max_value=168, 
    value=24) * 6 # in relative periods

# Sidebar links
st.sidebar.header('Links')
st.sidebar.markdown('[GitHub repository]()')


# Load data
df = load_data()

# Load model
model = load_model(model_dict[model_name])

# Get forecast
df_forecast = get_forecast(fd, fh, df, model)

# Plot graphs
fig = px.line(df_forecast, labels={'index': '', 'value': 'Power (kW)'})
st.plotly_chart(fig)

# Show metrics
metrics = get_metrics(df_forecast)
col1, col2, col3, col4, col5 = st.columns(5)
col2.metric(label='MAE', value=round(metrics['mae'], 2))
# col3.metric(label='RMSE', value=round(metrics['rmse'], 2))
col3.metric(label='MASE', value=round(metrics['mase'], 2))
# col5.metric(label='R²', value=round(metrics['r2'], 2))  # Display R-squared
# Original RMSE values for each model

rmse_values = np.array([
    467.41468802409025,  # Naive
    313.8237383371074,   # Linear
    311.6828717466473,   # Lasso
    567.5204355164487,   # KNN
    335.80936300179167,  # Decision Tree
    281.2799292956822,   # Random Forest
    244.95773895096232,  # LightGBM
    240.32260434487256   # CatBoost
])

# Apply min-max scaling to RMSE values to get them between 0 and 10
rmse_scaled = 10 * (rmse_values - rmse_values.min()) / (rmse_values.max() - rmse_values.min())

# Model names
models = ["Naive", "Linear", "Lasso", "KNN", "Decision Tree", "Random Forest", "LightGBM", "CatBoost"]

# Create a DataFrame for displaying the scaled RMSE and MASE values
comparison_df = pd.DataFrame({
    "Model": models,
    "Scaled RMSE": rmse_scaled.round(2),  })
# Streamlit display code
st.title("Model Performance Comparison")
st.write("Comparing scaled RMSE and MASE values for different models.")
# Display the DataFrame as a table
st.table(comparison_df)