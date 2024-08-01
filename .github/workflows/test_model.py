# test_model.py
import pytest

from model import model, X_test, y_test
from sklearn.metrics import mean_squared_error

def test_model_performance():
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    assert mse < 1.0
