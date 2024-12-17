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
import time

def main():
    # Initialize Dask client
    client = Client()
    print(client)

    # Step 1: Load Dataset
    file_path = "taxi_tripdata.csv"  # Replace with your dataset path
    print("Loading dataset with Dask...")
    df = dd.read_csv(file_path, assume_missing=True)

    # Step 2: Data Cleaning and Feature Engineering
    print("Preprocessing dataset...")

    # Clean and normalize column names
    df.columns = df.columns.str.strip().str.lower()

    # Check available columns
    print("Columns in the dataset:", df.columns)

    # Define feature and target column names
    trip_distance_col = 'trip_miles'
    fare_amount_col = 'base_passenger_fare'

    # Verify required columns exist
    if trip_distance_col not in df.columns or fare_amount_col not in df.columns:
        raise KeyError(f"Columns {trip_distance_col} or {fare_amount_col} not found in dataset")

    # Filter valid rows
    df = df[(df[trip_distance_col] > 0) & (df[fare_amount_col] > 0)]

    # Create a new feature: fare per mile
    df['fare_per_mile'] = df[fare_amount_col] / (df[trip_distance_col] + 1e-5)

    # Select relevant features
    features = [trip_distance_col, 'driver_pay', 'tips']
    target = fare_amount_col

    # Drop missing values
    df = df[features + [target]].dropna()

    # Repartition the data to ensure even distribution
    df = df.repartition(npartitions=10)

    # Step 3: Train-Test Split for Dask
    print("Splitting dataset...")
    X = df[features].to_dask_array(lengths=True)
    y = df[target].to_dask_array(lengths=True)

    X_train, X_test, y_train, y_test = dask_train_test_split(X, y, test_size=0.2, random_state=42)

    # Check shapes
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")

    # Step 4: Scale Features
    print("Scaling features...")
    scaler = StandardScaler()

    # Scale X_train and X_test
    X_train = scaler.fit_transform(X_train.compute())
    X_test = scaler.transform(X_test.compute())

    # Convert y_train and y_test to 1D arrays
    y_train = y_train.compute().flatten()
    y_test = y_test.compute().flatten()

    # Step 5: Train Model Using Dask
    print("Training Dask model...")
    dask_model = LinearRegression()
    dask_model.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = dask_model.predict(X_test)

    # Convert predictions to NumPy if necessary
    if hasattr(y_pred, 'compute'):
        y_pred = y_pred.compute()

    # Calculate MSE
    dask_mse = mean_squared_error(y_test, y_pred)
    print(f"Dask Model MSE: {dask_mse}")

    # Step 6: Train Model Using Pandas for Comparison
    print("Training traditional Pandas model...")
    start_time = time.time()

    # Load smaller data for Pandas
    pandas_df = pd.read_csv(file_path, nrows=50000)  
    pandas_df.columns = pandas_df.columns.str.strip().str.lower()

    # Filter valid rows
    pandas_df = pandas_df[(pandas_df[trip_distance_col] > 0) & (pandas_df[fare_amount_col] > 0)]
    pandas_df['fare_per_mile'] = pandas_df[fare_amount_col] / (pandas_df[trip_distance_col] + 1e-5)

    # Prepare features and target
    pandas_features = pandas_df[features]
    pandas_target = pandas_df[target]

    # Use sklearn's train_test_split
    X_train_pandas, X_test_pandas, y_train_pandas, y_test_pandas = sklearn_train_test_split(
        pandas_features, pandas_target, test_size=0.2, random_state=42, shuffle=True
    )

    # Train model
    pandas_model = SklearnLinearRegression()
    pandas_model.fit(X_train_pandas, y_train_pandas)
    pandas_y_pred = pandas_model.predict(X_test_pandas)

    # Calculate MSE
    end_time = time.time()
    pandas_mse = mean_squared_error(y_test_pandas, pandas_y_pred)
    print(f"Pandas Model MSE: {pandas_mse}")
    print(f"Pandas Training Time: {end_time - start_time:.2f} seconds")

    # Step 7: Visualization
    print("Visualizing results...")
    plt.figure(figsize=(10, 5))
    plt.scatter(y_test, y_pred, alpha=0.5, label='Dask Predictions')
    plt.xlabel("Actual Fare")
    plt.ylabel("Predicted Fare")
    plt.title("Dask Model: Predicted vs Actual Fare")
    plt.legend(loc='upper left')  # Explicitly set legend location
    plt.show()

    plt.bar(['Dask', 'Pandas'], [dask_mse, pandas_mse])
    plt.title("Model Mean Squared Error Comparison")
    plt.ylabel("MSE")
    plt.show()

    print("Pipeline completed.")

if __name__ == "__main__":
    main()