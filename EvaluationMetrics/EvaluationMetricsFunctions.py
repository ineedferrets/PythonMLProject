import numpy as np
import pandas as pd

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
# Measures the strength of the relationship between your linear model and the
# the dependent variables on a 0-1 scale.
def r2_score(actual, predicted):
    actual, predicted = np.array(actual), np.array(predicted)
    rss = np.sum((actual - predicted) ** 2)
    tss = np.sum((actual - np.mean(actual)) ** 2)
    return 1 - (rss / tss)

# Adjusted R-squared
# R2 always assumes that adding new features will always either increase or
# keep the score contant due to an summption that while adding more data,
# variance increases. Which isn't true if we add irrelevant feature data.
# This is where adjusted R-squared can help. Here:
# n = number of samples or data points
# k = number of independent variables or number of predictors/features
def adjusted_r2_score(R2, n, k):
    return 1 - ((1-R2)*(n-1)/(n-k-1))

# Classification Evaluation Metrics

# Confusion Matrix
# This is a table with combinations of predicted and actual variables. It
# compares the number of predictors for each class that are correct and
# those that are incorrect.
# The confusion matrix is NxN, where N is the number of classes or outputs.
def confusion_matrix(actual, predicted, label_name, margins=False):
    actual = pd.Series(actual, name='Actual')
    predicted = pd.Series(predicted, name='Predicted')
    df_confusion = pd.crosstab(actual, predicted, rownames=['Actual'], colnames=['Predicted'], margins=margins)
    df_confusion.rename(index=label_name, columns=label_name, inplace=True)
    return df_confusion

# We have four important values from these: True Positives (TP), False
# Positives (FP), True Negatives (TN), and False Negatives (FN).
def CM_parameters(cnf_matrix):
    TP = np.diag(cnf_matrix)
    FP = cnf_matrix.sum(axis=0) - TP
    FN = cnf_matrix.sum(axis=1) - TP
    TN = cnf_matrix.sum() - (FP + FN + TP)
    return TP.astype(float), TN.astype(float), FP.astype(float), FN.astype(float)

# We can use these to calculate important things:

# Accuracy
# Accuracy simply measures how often the classifier correctly predicts.
# This is the ratio of correct predictions to the total number of
# predictions. This can give you accuracy per class or a global accuracy:
def CM_accuracy(TP, TN, FP, FN, n):
    '''
    Parameters
    TP: List of number of true positives per class
    TN: List of number of true negatives per class
    FP: List of number of false positives per class
    FN: List of number of false negatives per class
    n: Total number of results
    Returns
    class_acc: Accuracy per class
    global_acc: Global accuracy
    '''
    class_acc = (TP+TN) / (TP+TN+FP+FN)
    global_acc = np.sum(TP) / n
    return class_acc, global_acc
    
