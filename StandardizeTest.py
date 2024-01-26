# Packages for data
import numpy as np
import matplotlib.pyplot as plt
import DataPreprocessingFunctions as dpf
import seaborn as sns

# Packages for unit testing
from sklearn.preprocessing import StandardScaler
import unittest

class TestStandardization(unittest.TestCase):

    def test_standardization(self):
        np.random.seed(42)
        X = np.random.normal(size=(1000,1), loc=1, scale=2)
        Z = dpf.standardize(X)
        
        # using sklearn to test our standardization function
        scaler = StandardScaler()
        scaler.fit(X)
        Zs = scaler.transform(X)
  
        self.assertTrue(np.array_equal(Z[:,0], Zs[:,0]), 'Standardization is not equal.')
        
if __name__ == '__main__':
    unittest.main()