import numpy as np

def onehot_encoding(X):
    '''
    Parameters:
    X: 1D array of labels of length "m"
    Return
    X_onehot: (m,d) one hot encoded matrix (one-hot of X) (where d is the
              number of unique values in X)
    indexes: dictionary containing the unique values of X and their mapping
             to the integer column
    '''
    indexes = {val:idx for idx, val in enumerate(np.unique(X))}
    y = np.array([indexes[s] for s in X])
    X_onehot = np.zeros((y.size, len(indexes)))
    X_onehot[np.arange(y.size), y] = 1
    return X_onehot, indexes  

X = np.array(['Apple', 'Mango', 'Apple', 'Apple', 'Orange', 'Mango'])

X_onehot, indexes = onehot_encoding(X)

print("Given dataset (X) =\n\n", X.reshape(-1,1))
print("\nMapping Dictionary = ", indexes)
print("\nOne Hot Encoding of X =\n\n", X_onehot)