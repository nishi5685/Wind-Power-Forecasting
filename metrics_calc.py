import pandas as pd
from sktime.performance_metrics.forecasting import (
    mean_absolute_error, 
    mean_squared_error, 
    mean_absolute_scaled_error
)
from sklearn.metrics import r2_score
def get_metrics(df_forecast: pd.DataFrame) -> dict:
    metrics = dict()

    # Mean Absolute Error
    metrics['mae'] = mean_absolute_error(
        y_true=df_forecast['true'].dropna(), 
        y_pred=df_forecast['forecast'].dropna()
    )

    # Root Mean Squared Error
    metrics['rmse'] = mean_squared_error(
        y_true=df_forecast['true'].dropna(), 
        y_pred=df_forecast['forecast'].dropna(), 
        square_root=True
    )

    # Mean Absolute Scaled Error
    metrics['mase'] = mean_absolute_scaled_error(
        y_true=df_forecast['true'].dropna(), 
        y_pred=df_forecast['forecast'].dropna(), 
        y_train=df_forecast['observed'].dropna()
    )

    # # R-squared Score
    # metrics['r2'] = r2_score(
    #     y_true=df_forecast['true'].dropna(), 
    #     y_pred=df_forecast['forecast'].dropna()
    # )
    
    return metrics
