import numpy as np

def minmax(X):
    Z = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
    return Z

X = np.array([1, 2, 3, 4, 5])
Z = minmax(X)

print("Original dataset (X) = ", X)
print("\nX (after min-max normalization) = ", Z)

X = np.array([[1, 7],
              [4, 3],
              [5, 2],
              [3, 6]])
Z = minmax(X)

print("Original dataset (X) = ", X)
print("\nX (after min-max normalization) = ", Z)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler();
scaler.fit(X)
Zs = scaler.transform(X)

print("\nX (after sklearn min-max normalization) = ", Zs)