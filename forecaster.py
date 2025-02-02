import pandas as pd
import numpy as np
from datetime import date
from sktime.base import BaseEstimator
from sktime.forecasting.base import ForecastingHorizon

def get_forecast(fd: date, fh: int, df: pd.DataFrame, model: BaseEstimator) -> pd.DataFrame:
    # Get observed data
    y_observed = df.loc[df.index < fd, ['power']]
    X_observed = df.loc[df.index < fd].drop('power', axis=1)

    # Update model with observed data
    model.update(y_observed, X_observed, update_params=False)

    # Check if fh is within available data range
    if fh <= 0 or fd not in df.index:
        raise ValueError("Invalid forecast horizon or forecast date.")

    # Get true target for the forecast horizon
    y_true = df.loc[fd:].head(fh)[['power']]
    if y_true.empty:
        raise ValueError("Forecast horizon is beyond available data range.")

    # Convert fh to sktime class for predict method
    fh_sktime = ForecastingHorizon(np.arange(1, min(fh, y_true.shape[0]) + 1), is_relative=True)

    # Prepare the forecast data
    X_forecast = df.loc[fd:].head(fh).drop('power', axis=1)
    y_forecast = model.predict(fh=fh_sktime, X=X_forecast)

    # Concatenate observed, true, and forecast data
    df_forecast = pd.concat([y_observed.tail(fh), y_true, y_forecast], axis=1)
    df_forecast.columns = ['observed', 'true', 'forecast']

    return df_forecast
