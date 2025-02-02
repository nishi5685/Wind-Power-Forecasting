import numpy as np

# Original RMSE values
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

# Apply min-max scaling to bring RMSE values between 0 and 10
rmse_scaled = 10 * (rmse_values - rmse_values.min()) / (rmse_values.max() - rmse_values.min())
rmse_scaled
