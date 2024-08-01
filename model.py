import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import os

# Generate some random data
X = np.random.rand(100, 1)
y = 3 * X.squeeze() + 2 + np.random.randn(100) * 0.5

# Set the artifact location to a valid directory
# Set up the tracking URI and artifact location
artifact_uri = os.getenv('MLFLOW_ARTIFACT_URI',
                         'file://' + os.path.join(os.getcwd(), 'mlruns'))

mlflow.set_tracking_uri(artifact_uri)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=42)

# Start an MLflow run
with mlflow.start_run():
    # Train a simple linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions
    predictions = model.predict(X_test)

    # Calculate mean squared error
    mse = mean_squared_error(y_test, predictions)

    # Log parameters and metrics
    mlflow.log_param("random_state", 42)
    mlflow.log_param("dataset_size", 200)
    mlflow.log_metric("mse", mse)

    # Log the model
    mlflow.sklearn.log_model(model, "model")

    print(f"Logged data and model in run: {mlflow.active_run().info.run_uuid}")
