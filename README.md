A Streamlit application for benchmarking data formats, comparing data processing libraries, and predicting stock prices using machine learning.

📋 Project Overview
This project:

Benchmarks CSV vs Parquet file formats
Compares Pandas vs Polars processing performance
Trains machine learning models to predict stock prices
Provides an interactive dashboard for stock price predictions

🚀 Getting Started
Prerequisites
Make sure you have Python 3.8+ installed. Then install the required packages:
pip install streamlit pandas polars numpy scikit-learn matplotlib joblib pyarrow

Dataset
The application requires the all_stocks_5yr.csv file

Open a terminal and navigate to the directory containing the script
Run the following command:

streamlit run stock_prediction_app.py

The application will open in your default web browser (typically at http://localhost:8501)


📄 File Structure
project_folder/
benchmarking.py    # Main application file
all_stocks_5yr.csv         # Original dataset 
all_stocks_5yr.parquet     # Converted dataset 
linear_regression_model.pkl # Saved model 
random_forest_model.pkl    # Saved model 
README.md                  # This file
