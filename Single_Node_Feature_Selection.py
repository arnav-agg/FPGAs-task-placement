import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LinearRegression as LR

from sklearn.feature_selection import RFE
from sklearn.svm import LinearSVC


def fitLR(X, y):
    lr = LR()
    y_pred = cross_val_predict(lr, X, y, cv=5)
    mse = np.mean((y_pred - y) ** 2)
    abse = np.mean(np.abs(y_pred - y))
    # print(y_pred.shape)
    print("LR: ", mse)
    print("LR: ", abse)
    print("")


if __name__ == '__main__':
    print("")
    # data = 'HeatmapData - 6 Features.csv'
    data = 'HeatmapData - 11 Features.csv'
    data = np.array(np.genfromtxt(data, delimiter=','))
    NumVars = np.shape(data)[1] - 4
    print("Number of variables:", NumVars)
    print(np.shape(data))
    dataFrame = pd.DataFrame(data=data)
    corrMat = dataFrame.corr()
    top_corr_features = corrMat.index
    plt.figure(figsize=(20, 20))
    g = sns.heatmap(dataFrame[top_corr_features].corr(), annot=True, cmap="RdYlGn")
    plt.show()

    # To train machine learning model with single node data. 'HeatmapData.csv' contains the data for P1 node.
    X = np.reshape(data[:, 0:NumVars], (np.size(data[:, 0]), NumVars))
    y1 = np.reshape(data[:, NumVars:(NumVars + 1)], np.size(data[:, 0]), 1)
    y2 = np.reshape(data[:, (NumVars + 1):(NumVars + 2)], np.size(data[:, 0]), 1)
    y3 = np.reshape(data[:, (NumVars + 2):(NumVars + 3)], np.size(data[:, 0]), 1)
    y4 = np.reshape(data[:, (NumVars + 3):(NumVars + 4)], np.size(data[:, 0]), 1)

    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    print("")
    print("P1:")
    fitLR(X, y1)
    print("P2:")
    fitLR(X, y2)
    print("P3:")
    fitLR(X, y3)
    print("P4:")
    fitLR(X, y4)

    '''
    print("")
    print("Other Feature Selection Stuff:")

    data = 'HeatmapData - Full.csv'
    data = np.array(np.genfromtxt(data, delimiter=','))
    NumVars = np.shape(data)[1] - 2
    X = np.reshape(data[:, 0:NumVars], (np.size(data[:, 0]), NumVars))
    y = np.reshape(data[:, NumVars:(NumVars + 1)], np.size(data[:, 0]), 1)

    model = LR()
    rfe = RFE(model, 6)
    rfe = rfe.fit(X, y)
    print(rfe.support_)
    print(rfe.ranking_)
    print(len(rfe.ranking_))
    '''
