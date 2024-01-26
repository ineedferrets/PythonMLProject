import numpy as np

# Regression evaluation metrics

# Mean Absolute Error (MAE)
# Advantages: - The error is in the same unit as the output.
#             - It is most robust to outliers.
# Disadvantages: The graph of MAE is not differentiable, optimizers like
#                gradient descent have to be applied.
def mae(actual, predicted):
    actual, predicted = np.array(actual), np.array(predicted)
    return np.mean(np.abs(actual - predicted))

# Mean Absolute Percentage Error (MAPE)
# Advantages: - One of the most widely used measures of forecast accuracy
#             due to its advantages of scale-independency and interpretability.
# Disadvantages: - It produces infinite of undefined values for zero or
#                or close-to-zero actual values
def mape(actual, predicted):
    actual, predicted = np.array(actual), np.array(predicted)
    return np.mean(np.abs((actual - predicted) / actual))

# Mean Squared Error (MSE)
# Advantages: - The graph is differentiable, so it can easily be used as a
#             loss function.
# Disadvantages: - The value given after calculating MSE is a squared unit of
#                  the output.
#                - This penalises outliers the most and is not robust to them.
def mse(actual, predicted):
    actual, predicted = np.array(actual), np.array(predicted)
    return np.mean((actual - predicted) ** 2)

# Root Mean Squared Error (RMSE)
# Advantages: - The output value you get is the same unit as the required
#               output variable, making the interpretation of loss easier.
# Disadvantages: - It is not robust to outliers.
def rmse(actual, predicted):
    actual, predicted = np.array(actual), np.array(predicted)
    return np.sqrt(np.mean((actual - predicted) ** 2))

# Coefficient of Determination (R-squared or R2)
# 