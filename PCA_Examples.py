import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

url = "http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
df = pd.read_csv(url, names=['sepal length', 'sepal width', 'petal length', 'petal width', 'target'])

# print(df.head())

features = ['sepal length', 'sepal width', 'petal length', 'petal width']
X = df.loc[:, features].values  # (150,4)
y = df.loc[:, ['target']].values  # (150, 1)

# Standardize Data, Mean of 0 STD of 1
scaler = StandardScaler()
X = scaler.fit_transform(X)
# print(X[:5])  # head
X_Save = X[0]
# print(X[-5:])  # tail

'''
pca = PCA(0.95)
pca.fit(X)
print("Number of principal components:", pca.n_components_)
Data = pca.transform(X)
'''

pca = PCA(n_components=2)
pca.fit(X)

# ----------------------------------------------------------------------------------------------------------------------
# Should look into this. There is an array for each principal component, where the values in the array correspond to the
# weights for the original features/variables. To get the value of the principal component, you take the dot product of
# the array for the principal component with the normalized data for each point. (Normalized data: print(X[:1]), array:
# pca.components_, Values: finalDF)

# print("Test:", pca.components_)
Comp_Save = pca.components_
principalComp = pca.transform(X)
print(principalComp[:5])

principalData = pd.DataFrame(data=principalComp, columns=['principal component 1', 'principal component 2'])

finalDF = pd.concat([principalData, df[['target']]], axis=1)

print(finalDF.head())
print("")
print("Testing to check first value of both primary components:")
print(np.dot(X_Save, Comp_Save[0]))
print(np.dot(X_Save, Comp_Save[1]))


fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('Principal Component 1', fontsize=15)
ax.set_ylabel('Principal Component 2', fontsize=15)
ax.set_title('2 Component PCA', fontsize=20)

targets = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
colors = ['r', 'g', 'b']
for target, color in zip(targets, colors):
    indicesToKeep = finalDF['target'] == target
    ax.scatter(finalDF.loc[indicesToKeep, 'principal component 1'], finalDF.loc[indicesToKeep, 'principal component 2'],
               c=color, s=50)
    ax.legend(targets)
    ax.grid()

plt.show()

VRatio = pca.explained_variance_ratio_
print("")
print("These components account for", sum(VRatio), "of the variance of the dataset")


