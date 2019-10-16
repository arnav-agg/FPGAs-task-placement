import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression as LR
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_predict


def fitme(X, y):
    svr = SVR(kernel='linear', C=27, epsilon=0.5)
    # svr = SVR(kernel='linear', degree=3, gamma='auto', C=100, epsilon=0.01)

    y_pred = cross_val_predict(svr, X, y, cv=10)
    mse = np.sqrt(np.mean((y_pred - y) ** 2))
    abse = np.mean(np.abs(y_pred - y))
    # print(y_pred.shape)
    print("MSE: ", mse)
    print("MAE: ", abse)
    print("")


if __name__ == '__main__':
    # Data parsing
    data = 'HeatmapData - 9 Features - LU&CU Node 2.csv'
    data = np.array(np.genfromtxt(data, delimiter=','))
    NumVars = np.shape(data)[1] - 1

    '''
    # X_full = np.reshape(data[:-18, 0:NumVars], (np.size(data[:-18, 0]), NumVars))
    # y_full = np.reshape(data[:-18, NumVars:(NumVars + 1)], np.size(data[:-18, 0]), 1)
    X_full = np.reshape(data[:, 0:NumVars], (np.size(data[:, 0]), NumVars))
    y_full = np.reshape(data[:, NumVars:(NumVars + 1)], np.size(data[:, 0]), 1)

    # perms = np.random.permutation(np.shape(X_full)[0])
    # X_full = X_full[perms]
    # y_full = y_full[perms]

    scaler = StandardScaler()
    scaler.fit(X_full)
    X_full = scaler.transform(X_full)

    print("P2:")
    fitme(X_full, y_full)
    '''

    # Used to choose the specific set of observations for training/testing
    X_train = np.reshape(data[:-5, 0:NumVars], (np.size(data[:-5, 0]), NumVars))
    y_train = np.reshape(data[:-5, NumVars:(NumVars + 1)], np.size(data[:-5, 0]), 1)

    X_test = np.reshape(data[-5:, 0:NumVars], (np.size(data[-5:, 0]), NumVars))
    y_test = np.reshape(data[-5:, NumVars:(NumVars + 1)], np.size(data[-5:, 0]), 1)

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    print(np.shape(X_train), np.shape(y_train))
    print(np.shape(X_test), np.shape(y_test))

    model = SVR(kernel='linear', C=27, epsilon=0.5)
    model.fit(X_train, y_train)

    print("Pred:", model.predict(X_test))
    print("Actual:", y_test)

    '''
    # SVR - Linear
    C_r = [0.001, 0.01, 0.1, 1, 5, 10, 20, 50, 80, 100, 200, 300]
    E_r = [0.001, 0.01, 0.1, 1, 5, 10, 20, 50, 80, 100, 200, 300]
    C_r = range(10, 30)
    E_r = np.linspace(0.1, 0.7, 7)

    param_grid = dict(C=C_r, epsilon=E_r)

    svr = SVR(kernel='linear', C=27, epsilon=0.5)

    grid = GridSearchCV(svr, param_grid, cv=5, n_jobs=-1, scoring='neg_mean_squared_error')
    grid.fit(X_full, y_full)

    print("Parameter used:", grid.best_params_)
    print("Best score:", grid.best_score_)
    '''