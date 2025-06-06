import os
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from pmdarima import auto_arima

# ────────────────────────────────────────────────────────────────────────────────
# Suppress warnings (optional)
warnings.filterwarnings("ignore")

# ─────────── USER CONFIG ────────────────────────────────────────────────────────
# 1) Choose a single family to model (replace with an exact family string from train.csv)
FAMILY = "YOUR_FAMILY_NAME_HERE"

# 2) File paths (adjust if needed)
TRAIN_CSV = "../Data/DM/train.csv"   # Must contain: [id, date, store_nbr, family, sales, onpromotion]
TEST_CSV  = "../Data/DM/test.csv"    # Must contain: [id, date, store_nbr, family, onpromotion]

# 3) Output filenames for this family
PERF_CSV     = f"performance_{FAMILY.replace(' ', '_')}.csv"
FORECAST_CSV = f"forecast_{FAMILY.replace(' ', '_')}.csv"

# 4) Hold-out size (set to 0 to skip hold-out)
N_VALID = 14

# 5) Time-series frequency
FREQ = "D"  # daily

# 6) auto_arima arguments
AUTO_ARIMA_ARGS = {
    "seasonal": True,
    "m": 7,  # weekly seasonality
    "start_p": 0, "start_q": 0, "max_p": 5, "max_q": 5,
    "start_P": 0, "start_Q": 0, "max_P": 2, "max_Q": 2,
    "d": None, "D": None,
    "trace": False,
    "error_action": "ignore",
    "suppress_warnings": True,
    "stepwise": True,
    "information_criterion": "aic"
}

print(f"→ Modeling only family = '{FAMILY}'\n")

# ────────────────────────────────────────────────────────────────────────────────
# Helper: ADF stationarity test
def test_stationarity(ts, name="series"):
    """
    Perform the Augmented Dickey-Fuller test on a 1D series `ts`.
    Prints ADF statistic, p-value, and critical values.
    """
    result = adfuller(ts)
    print("  ─" * 30)
    print(f"  ADF Statistic ({name}): {result[0]:.5f}")
    print(f"  p-value:            {result[1]:.5f}")
    for key, val in result[4].items():
        print(f"  Critical Value ({key}): {val:.5f}")
    if result[1] < 0.05:
        print("  → The series IS stationary (reject H₀).")
    else:
        print("  → The series is NOT stationary (fail to reject H₀).")
    print("  ─" * 30)

# ────────────────────────────────────────────────────────────────────────────────
# 1) Load train.csv and filter by FAMILY
df_train_all = pd.read_csv(TRAIN_CSV, parse_dates=["date"])
df_train = df_train_all[df_train_all["family"] == FAMILY].copy()
if df_train.empty:
    raise ValueError(f"No rows found for family '{FAMILY}' in {TRAIN_CSV}")

# 2) Load test.csv and filter by FAMILY
df_test_all = pd.read_csv(TEST_CSV, parse_dates=["date"])
df_test = df_test_all[df_test_all["family"] == FAMILY].copy()
if df_test.empty:
    print(f"Warning: No rows found for family '{FAMILY}' in {TEST_CSV}. Forecasting will be skipped.")

# 3) Aggregate train by date (sum sales + onpromotion across all stores)
df_train_agg = (
    df_train
    .groupby("date")[["sales", "onpromotion"]]
    .sum()
    .rename(columns={"onpromotion": "onpromo"})
    .sort_index()
)

# 4) Convert to PeriodIndex (daily) and fill missing dates with zeros
df_train_agg.index = pd.DatetimeIndex(df_train_agg.index).to_period(FREQ)
df_train_agg = df_train_agg.asfreq(FREQ)
df_train_agg["sales"]   = df_train_agg["sales"].fillna(0).astype(float)
df_train_agg["onpromo"] = df_train_agg["onpromo"].fillna(0).astype(float)

# 5) Remove rows where sales == 0
orig_len = len(df_train_agg)
df_train_agg = df_train_agg[df_train_agg["sales"] != 0].copy()
print(f"Removed {orig_len - len(df_train_agg)} zero-sales rows → {len(df_train_agg)} remaining")

# 6) Check if enough points remain
if len(df_train_agg) < (N_VALID + 1):
    raise ValueError(f"Not enough data points ({len(df_train_agg)}) for hold-out of {N_VALID} days.")

# 7) Split hold-out (if N_VALID > 0)
if N_VALID > 0:
    df_valid_holdout = df_train_agg.iloc[-N_VALID:]
    df_train_series  = df_train_agg.iloc[:-N_VALID]
    print(f"Held out last {N_VALID} days for validation.")
else:
    df_valid_holdout = None
    df_train_series  = df_train_agg

# 8) Stationarity test on raw sales
print("\nRunning ADF test on training 'sales' series:")
test_stationarity(df_train_series["sales"], name="sales")

# 9) Log-transform the training sales
df_train_series["y_log"] = np.log1p(df_train_series["sales"])

# 10) Prepare exogenous arrays and scale them
exog_train_raw = df_train_series["onpromo"].values.reshape(-1, 1)
if df_valid_holdout is not None:
    exog_valid_raw = df_valid_holdout["onpromo"].values.reshape(-1, 1)

scaler = StandardScaler()
exog_train_scaled = scaler.fit_transform(exog_train_raw).ravel()
exog_train = pd.Series(exog_train_scaled, index=df_train_series.index)

if df_valid_holdout is not None:
    exog_valid_scaled = scaler.transform(exog_valid_raw).ravel()
    exog_valid = pd.Series(exog_valid_scaled, index=df_valid_holdout.index)
else:
    exog_valid = None

# 11) Run auto_arima for order selection
print("\nRunning auto_arima for hyperparameter tuning:")
mi = auto_arima(
    df_train_series["y_log"],
    exogenous=exog_train.values.reshape(-1, 1),
    **AUTO_ARIMA_ARGS
)
order_opt          = mi.order
seasonal_order_opt = mi.seasonal_order
print(f"Selected order = {order_opt}, seasonal_order = {seasonal_order_opt}")

# 12) Fit SARIMAX on the training series
print("\nFitting SARIMAX on log-transformed series with scaled exogenous:")
model = SARIMAX(
    df_train_series["y_log"],
    exog=exog_train,
    order=order_opt,
    seasonal_order=seasonal_order_opt,
    enforce_stationarity=False,
    enforce_invertibility=False
)
sarimax_fit = model.fit(disp=False)
print(sarimax_fit.summary())

# 13) Evaluate hold-out performance (if applicable)
if df_valid_holdout is not None:
    print("\nComputing hold-out metrics:")
    pred_log_valid = sarimax_fit.get_forecast(
        steps=N_VALID,
        exog=exog_valid.values.reshape(-1, 1)
    ).predicted_mean
    pred_valid   = np.expm1(pred_log_valid)
    actual_valid = df_valid_holdout["sales"].values

    rmse_val = np.sqrt(mean_squared_error(actual_valid, pred_valid))
    mae_val  = mean_absolute_error(actual_valid, pred_valid)
    r2_val   = r2_score(actual_valid, pred_valid)
    print(f"Hold-out metrics:  RMSE = {rmse_val:.2f},  MAE = {mae_val:.2f},  R² = {r2_val:.3f}")

    # Save hold-out performance to CSV
    perf_df = pd.DataFrame([{
        "family": FAMILY,
        "RMSE": rmse_val,
        "MAE": mae_val,
        "R2": r2_val
    }])
    perf_df.to_csv(PERF_CSV, index=False)
    print(f"Hold-out performance saved to {PERF_CSV}")

# 14) Forecast on test dates (if test data exists)
if not df_test.empty:
    # Aggregate test exogenous by date
    df_test["date"] = pd.to_datetime(df_test["date"])
    agg_exog_test = (
        df_test
        .groupby("date")[["onpromotion"]]
        .sum()
        .rename(columns={"onpromotion": "onpromo"})
    )
    agg_exog_test.index = pd.DatetimeIndex(agg_exog_test.index).to_period(FREQ)
    agg_exog_test = agg_exog_test.asfreq(FREQ, fill_value=0)

    # Scale test exogenous
    exog_test_raw   = agg_exog_test["onpromo"].values.reshape(-1, 1)
    exog_test_scaled = scaler.transform(exog_test_raw).ravel()

    # Forecast
    h = len(agg_exog_test)
    print(f"\nForecasting {h} future points on test set:")
    pred_log_test  = sarimax_fit.get_forecast(
        steps=h,
        exog=exog_test_scaled.reshape(-1, 1)
    ).predicted_mean
    pred_sales_test = np.expm1(pred_log_test)

    # Build a date-level forecast DataFrame
    df_forecast_dates = pd.DataFrame({
        "date": agg_exog_test.index.to_timestamp(),
        "predicted_sales": pred_sales_test
    })

    # Merge date-level predictions back onto every row of df_test
    df_test = df_test.merge(
        df_forecast_dates,
        left_on="date",
        right_on="date",
        how="left"
    )

    # Prepare final forecast output (one row per original test id)
    df_forecast_output = pd.DataFrame({
        "id": df_test["id"].values,
        "date": df_test["date"].values,
        "store_nbr": df_test["store_nbr"].values,
        "family": FAMILY,
        "predicted_sales": df_test["predicted_sales"].values
    })

    # Save to CSV
    df_forecast_output.to_csv(FORECAST_CSV, index=False)
    print(f"Forecasts saved to {FORECAST_CSV}")
    print(df_forecast_output.head(10))
else:
    print("\nNo test data for this family; skipping forecast step.")