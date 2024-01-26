import numpy as np

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

X = np.array([['Apple', 'Tomato', 8],
              ['Mango', 'Tomato', 3],
              ['Apple', 'Carrot', 7],
              ['Orange', 'Potato', 4],
              ['Papaya', 'Carrot', 6],
              ['Banana', 'Tomato', 7],
              ['Orange', 'Potato', 7]])

Y = X[:, :-1]
Y_label_enc, idx_list = label_encoding(Y)

print("Categorical dataset (Y) =\n\n", Y)
print("\nMapping List of Dictionaries =\n\n", idx_list)
print("\nLabel Encoding of Y =\n\n", Y_label_enc)
