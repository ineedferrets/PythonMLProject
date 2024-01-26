import numpy as np

def standardize(X):
    Z = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    return Z

def minmax(X):
    Z = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
    return Z

def label_encoding(Y):
    '''
    Parameters:
    Y: (m,d) shape matrix with categorical data
    Return result: label encoded data of Y
    idx_list: list of all the dictionaries containing the unique
              values of the columns and their mapping to the
              integer.
    '''
    idx_list = []
    result = []
    for col in range(Y.shape[1]):
        indexes = {val: idx for idx, val in enumerate(np.unique(Y[:,col]))}
        result.append([indexes[s] for s in Y[:,col]])
        idx_list.append(indexes)
    return np.array(result).T, idx_list

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
    X_onehot[np.arrange(y.size), y] = 1
    return X_onehot, indexes    