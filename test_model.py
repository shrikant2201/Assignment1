# import sys
# import os
# from sklearn.metrics import mean_squared_error
# from model import model, X_test, y_test
# # Add the directory containing the script to the system path
# sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))


# def test_dummy():
#     assert 1 + 1 == 2


# def test_model_performance():
#     predictions = model.predict(X_test)
#     mse = mean_squared_error(y_test, predictions)
#     assert mse < 1.0


# # Run the test function
# if __name__ == "__main__":
#     test_model_performance()
#     print("Test passed")
