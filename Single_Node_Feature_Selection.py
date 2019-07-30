import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LinearRegression as LR


def fitLR(X, y):
    lr = LR()
    y_pred = cross_val_predict(lr, X, y, cv=10)
    mse = np.mean((y_pred - y) ** 2)
    abse = np.mean(np.abs(y_pred - y))
    # print(y_pred.shape)
    print("")
    print("LR: ", mse)
    print("LR: ", abse)
    print("\n")


if __name__ == '__main__':
    data = 'HeatmapData.csv'
    data = np.array(np.genfromtxt(data, delimiter=','))
    print(np.shape(data))
    NumVars = np.shape(data)[1] - 2
    dataFrame = pd.DataFrame(data=data)
    corrMat = dataFrame.corr()
    top_corr_features = corrMat.index
    print(np.shape(corrMat))
    plt.figure(figsize=(20, 20))
    g = sns.heatmap(dataFrame[top_corr_features].corr(), annot=True, cmap="RdYlGn")
    plt.show()

    # To train machine learning model with single node data. 'HeatmapData.csv' contains the data for P1 node.
    X = np.reshape(data[:, 0:NumVars], (np.size(data[:, 0]), NumVars))
    y = np.reshape(data[:, NumVars:(NumVars + 1)], np.size(data[:, 0]), 1)

    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    fitLR(X, y)




