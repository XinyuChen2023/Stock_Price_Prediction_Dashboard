import streamlit as st
import pandas as pd
import polars as pl
import time
import os
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# File paths
csv_file = "all_stocks_5yr.csv"
parquet_file = "all_stocks_5yr.parquet"

# ✅ Part 1: Benchmarking CSV vs Parquet
st.title("📊 Stock Price Prediction Dashboard")
st.header("🚀 Benchmark Results")

@st.cache_data
def load_data():
    """Load dataset and benchmark CSV vs Parquet performance."""
    csv_read_time, csv_write_time, parquet_read_time, parquet_write_time = None, None, None, None
    csv_size, parquet_size = None, None

    if not os.path.exists(parquet_file):
        # Read CSV
        start = time.time()
        df_pandas = pd.read_csv(csv_file, parse_dates=["date"])
        end = time.time()
        csv_read_time = round(end - start, 2)

        # Write Parquet
        start = time.time()
        df_pandas.to_parquet(parquet_file, engine='pyarrow', compression='snappy')
        end = time.time()
        parquet_write_time = round(end - start, 2)
    
    # Read Parquet
    start = time.time()
    df_pandas = pd.read_parquet(parquet_file, engine='pyarrow')
    end = time.time()
    parquet_read_time = round(end - start, 2)

    # Compare file sizes
    if os.path.exists(csv_file):
        csv_size = round(os.path.getsize(csv_file) / (1024 ** 2), 2)
    parquet_size = round(os.path.getsize(parquet_file) / (1024 ** 2), 2)

    return df_pandas, csv_read_time, csv_write_time, parquet_read_time, parquet_write_time, csv_size, parquet_size

df_pandas, csv_read_time, csv_write_time, parquet_read_time, parquet_write_time, csv_size, parquet_size = load_data()

if csv_read_time is not None:
    st.metric("CSV Read Time (sec)", csv_read_time)
    st.metric("CSV Write Time (sec)", csv_write_time)
    st.metric("CSV File Size (MB)", csv_size)

st.metric("Parquet Read Time (sec)", parquet_read_time)
if parquet_write_time is not None:
    st.metric("Parquet Write Time (sec)", parquet_write_time)
st.metric("Parquet File Size (MB)", parquet_size)

# ✅ Pandas vs Polars Performance
st.subheader("📊 Pandas vs Polars Execution Time")

# Measure Pandas Performance
start = time.time()
df_pandas["MFI_14"] = (df_pandas["high"] + df_pandas["low"] + df_pandas["close"]) / 3
df_pandas["MACD"] = df_pandas["close"].ewm(span=12, adjust=False).mean() - df_pandas["close"].ewm(span=26, adjust=False).mean()
df_pandas["MACD_Signal"] = df_pandas["MACD"].ewm(span=9, adjust=False).mean()
pandas_exec_time = round(time.time() - start, 2)

# Measure Polars Performance
df_polars = pl.read_parquet(parquet_file)
start = time.time()
df_polars = df_polars.with_columns([
    ((pl.col("high") + pl.col("low") + pl.col("close")) / 3).alias("MFI_14"),
    (pl.col("close").ewm_mean(span=12) - pl.col("close").ewm_mean(span=26)).alias("MACD"),
])
df_polars = df_polars.with_columns([
    pl.col("MACD").ewm_mean(span=9).alias("MACD_Signal")
])
polars_exec_time = round(time.time() - start, 2)

st.metric("Pandas Execution Time (sec)", pandas_exec_time)
st.metric("Polars Execution Time (sec)", polars_exec_time)

# ✅ Part 2: Feature Engineering & Model Training
st.header("🔬 Feature Engineering & Model Training")

# Drop NaNs before training
df_clean = df_pandas.dropna(subset=["MFI_14", "MACD", "MACD_Signal", "close"])

# Train-Test Split
X = df_clean[["MFI_14", "MACD", "MACD_Signal"]]
y = df_clean["close"].shift(-1)

# Align X and y lengths by dropping the last row from X
X = X.iloc[:-1]
y = y.iloc[:-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=42)

# Train Models
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Save Models
joblib.dump(lr_model, "linear_regression_model.pkl")
joblib.dump(rf_model, "random_forest_model.pkl")
st.success("✅ Models trained & saved!")

# ✅ Part 3: Stock Price Prediction
st.header("📈 Stock Price Prediction")

# Dropdown to select company
company_tickers = df_pandas["name"].unique()
selected_ticker = st.selectbox("Select a Company:", company_tickers)

# Filter data for the selected company
company_data = df_pandas[df_pandas["name"] == selected_ticker].copy()

# Calculate features if they don't exist in the filtered data
if "MFI_14" not in company_data.columns:
    company_data["MFI_14"] = (company_data["high"] + company_data["low"] + company_data["close"]) / 3
    company_data["MACD"] = company_data["close"].ewm(span=12, adjust=False).mean() - company_data["close"].ewm(span=26, adjust=False).mean()
    company_data["MACD_Signal"] = company_data["MACD"].ewm(span=9, adjust=False).mean()

# Make Predictions
features = ["MFI_14", "MACD", "MACD_Signal"]
company_features = company_data[features].dropna()

if not company_features.empty:
    lr_predictions = lr_model.predict(company_features)
    rf_predictions = rf_model.predict(company_features)

    # Plot results
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(company_data["date"].iloc[-len(lr_predictions):], company_data["close"].iloc[-len(lr_predictions):], label="Actual Price", color="blue")
    ax.plot(company_data["date"].iloc[-len(lr_predictions):], lr_predictions, label="Linear Regression", color="green")
    ax.plot(company_data["date"].iloc[-len(lr_predictions):], rf_predictions, label="Random Forest", color="red")
    ax.set_xlabel("Date")
    ax.set_ylabel("Stock Price")
    ax.legend()
    st.pyplot(fig)

    # Display Prediction DataFrame
    prediction_df = pd.DataFrame({
        'Date': company_data["date"].iloc[-len(lr_predictions):].reset_index(drop=True),
        'Actual Price': company_data["close"].iloc[-len(lr_predictions):].reset_index(drop=True),
        'Linear Regression Prediction': lr_predictions,
        'Random Forest Prediction': rf_predictions
    })

    st.subheader("Prediction Results")
    st.dataframe(prediction_df.tail(10))