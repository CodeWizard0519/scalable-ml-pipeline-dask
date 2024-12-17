import subprocess
import sys
import os
import math
import time
import logging
import argparse

# Install Missing Packages
def install_packages():
    required_packages = [
        "dask[complete]",
        "dask-ml",
        "scikit-learn",
        "pandas",
        "matplotlib",
        "tqdm"
    ]
    for package in required_packages:
        try:
            __import__(package.split("[")[0])
        except ImportError:
            print(f"🔧 Installing missing package: {package}")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package, "--quiet"])

install_packages()

from dask.distributed import Client
import dask.dataframe as dd
from dask_ml.model_selection import train_test_split as dask_train_test_split
from dask_ml.linear_model import LinearRegression
from sklearn.model_selection import train_test_split as sklearn_train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression as SklearnLinearRegression
from sklearn.metrics import mean_squared_error
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# Setup Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger()

# Helper Functions
def print_stage(message):
    logger.info(f"\n{'='*50}\n✅ {message}\n{'='*50}")

def save_plot(fig, filename):
    filepath = os.path.join("outputs", filename)
    os.makedirs("outputs", exist_ok=True)
    fig.savefig(filepath)
    logger.info(f"📊 Visualization saved: {filepath}")

def save_results(results, filename="results.txt"):
    with open(filename, "w") as f:
        for line in results:
            f.write(line + "\n")
    logger.info(f"📝 Results saved to: {filename}")

def print_performance_table(dask_mse, pandas_mse, pandas_time):
    print("\n" + "="*50)
    print("📊 **Performance Comparison**")
    print("="*50 + "\n")
    
    # Table header
    print(f"| {'**Metric**':<20} | {'**Dask Pipeline**':<20} | {'**Pandas Pipeline**':<20} |")
    print(f"|{'-'*21}|{'-'*21}|{'-'*21}|")
    
    # Table content
    print(f"| {'**MSE**':<20} | {dask_mse:<20.2f} | {pandas_mse:<20.2f} |")
    print(f"| {'**Training Time**':<20} | {'Distributed':<20} | {pandas_time:<20.2f} seconds |")
    print("\n" + "="*50)

def load_and_preprocess_data(file_path):
    try:
        logger.info("Loading dataset...")
        file_size = os.path.getsize(file_path) / (1024 ** 2)
        num_partitions = max(1, math.ceil(file_size / 50))
        
        logger.info(f"Dataset size: {file_size:.2f} MB, using {num_partitions} partitions.")
        with tqdm(desc="Reading Dataset", total=1) as pbar:
            df = dd.read_csv(file_path, assume_missing=True)
            pbar.update(1)
        df.columns = df.columns.str.strip().str.lower()

        # Clean data
        df = df[(df['trip_miles'] > 0) & (df['base_passenger_fare'] > 0)]
        df['fare_per_mile'] = df['base_passenger_fare'] / (df['trip_miles'] + 1e-5)
        df = df[['trip_miles', 'driver_pay', 'tips', 'base_passenger_fare']].dropna()
        df = df.repartition(npartitions=num_partitions)
        logger.info("✅ Data loaded and preprocessed successfully!")
        return df
    except Exception as e:
        logger.error(f"Error during data loading: {e}")
        raise

def scale_features(X_train, X_test):
    logger.info("⏳ Scaling features...")
    with tqdm(total=2, desc="Scaling Progress") as pbar:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train.compute())
        pbar.update(1)
        X_test = scaler.transform(X_test.compute())
        pbar.update(1)
    return X_train, X_test

def train_dask_model(X_train, y_train, X_test, y_test):
    logger.info("⏳ Training Dask model...")
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    logger.info(f"✅ Dask Model MSE: {mse:.2f}")
    return mse, y_pred

def train_pandas_model(file_path):
    logger.info("⏳ Training Pandas model...")
    start_time = time.time()
    df = pd.read_csv(file_path, nrows=50000)
    df.columns = df.columns.str.strip().str.lower()
    df = df[(df['trip_miles'] > 0) & (df['base_passenger_fare'] > 0)]
    df['fare_per_mile'] = df['base_passenger_fare'] / (df['trip_miles'] + 1e-5)

    features = df[['trip_miles', 'driver_pay', 'tips']]
    target = df['base_passenger_fare']
    X_train, X_test, y_train, y_test = sklearn_train_test_split(features, target, test_size=0.2, random_state=42)

    model = SklearnLinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    training_time = time.time() - start_time
    logger.info(f"✅ Pandas Model MSE: {mse:.2f}, Training Time: {training_time:.2f} seconds")
    return mse, training_time

def main():
    parser = argparse.ArgumentParser(description="Dask ML Pipeline")
    parser.add_argument("--data", type=str, required=True, help="Path to the dataset file")
    args = parser.parse_args()

    if not os.path.isfile(args.data):
        logger.error(f"❌ File '{args.data}' does not exist.")
        sys.exit(1)

    print_stage("Initializing Dask Client")
    client = Client()
    logger.info(f"🚀 Dask Client: {client}")

    df = load_and_preprocess_data(args.data)
    X = df[['trip_miles', 'driver_pay', 'tips']].to_dask_array(lengths=True)
    y = df['base_passenger_fare'].to_dask_array(lengths=True)

    X_train, X_test, y_train, y_test = dask_train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_test = scale_features(X_train, X_test)
    y_train, y_test = y_train.compute().flatten(), y_test.compute().flatten()

    dask_mse, y_pred_dask = train_dask_model(X_train, y_train, X_test, y_test)
    pandas_mse, pandas_time = train_pandas_model(args.data)

    print_performance_table(dask_mse, pandas_mse, pandas_time)

    logger.info("✅ Pipeline Completed Successfully 🎉")

if __name__ == "__main__":
    main()