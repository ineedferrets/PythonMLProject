import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def standardize(X):
    Z = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    return Z

np.random.seed(42)
X = np.random.normal(size=(1000,1), loc=1, scale=2)

Z = standardize(X)

# plot
sns.kdeplot(X[:,0], label="X (without standardization)", color='k')
sns.kdeplot(Z[:,0], label="X (with standardization)", color='r')
plt.title("Distribution of X")
plt.legend()
plt.show()

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X)
Zs = scaler.transform(X)

# plot comparing standardization
sns.kdeplot(Z[:,0], label="standardization - our function", color='k')
sns.kdeplot(Zs[:,0], label="standardization - sklearn", color='r')
plt.title("Standardization (our function woth sklearn)")
plt.legend()
plt.show()