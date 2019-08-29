import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LinearRegression as LR
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

import xgboost as xgb

from sklearn.model_selection import GridSearchCV

import warnings

import time


def fitLR(X, y):
    lr = LR()

    rf = RandomForestRegressor(n_estimators=37, max_depth=14, min_samples_split=2, min_samples_leaf=1,
                               max_features=1, random_state=1)

    lasso = Lasso(alpha=0.07, max_iter=10000, tol=1e-3)
    ridge = Ridge(alpha=0.1, max_iter=10000)
    nn = KNeighborsRegressor(n_neighbors=6, p=2)

    mlp = MLPRegressor(hidden_layer_sizes=(3, 5), learning_rate_init=0.12, random_state=1, batch_size=7, max_iter=1000)

    grad = GradientBoostingRegressor(loss='lad', learning_rate=0.15, min_samples_split=2, n_estimators=90, max_depth=3,
                                     min_samples_leaf=3, max_features=8, subsample=0.8, random_state=0)

    xg_reg = xgb.XGBRegressor(learning_rate=0.1, n_estimators=50, max_depth=5, min_child_weight=1, gamma=1.5,
                              subsample=0.3, scale_pos_weight=1, colsample_bytree=0.65, verbosity=0, reg_alpha=1.3,
                              reg_lambda=0)

    svr = SVR(kernel='linear', C=18, epsilon=0.3)
    # svr = SVR(kernel='linear', degree=3, gamma='auto', C=100, epsilon=0.01)

    y_pred = cross_val_predict(rf, X, y, cv=5)
    mse = np.mean((y_pred - y) ** 2)
    abse = np.mean(np.abs(y_pred - y))
    # print(y_pred.shape)
    print("MSE: ", mse)
    print("MAE: ", abse)
    print("")


if __name__ == '__main__':
    warnings.simplefilter("ignore")

    print("")
    data = 'HeatmapData - 9 Features.csv'
    # data = 'HeatmapData.csv'
    data = np.array(np.genfromtxt(data, delimiter=','))
    NumVars = np.shape(data)[1] - 4

    '''
    # Heatmap Stuff
    print("Number of variables:", NumVars)
    print(np.shape(data))
    dataFrame = pd.DataFrame(data=data)
    corrMat = dataFrame.corr()
    top_corr_features = corrMat.index
    plt.figure(figsize=(20, 20))
    g = sns.heatmap(dataFrame[top_corr_features].corr(), annot=True, cmap="RdYlGn")
    plt.show()
    '''

    # To train machine learning model with single node data. 'HeatmapData.csv' contains the data for P1 node.
    X = np.reshape(data[:, 0:NumVars], (np.size(data[:, 0]), NumVars))
    y1 = np.reshape(data[:, NumVars:(NumVars + 1)], np.size(data[:, 0]), 1)
    y2 = np.reshape(data[:, (NumVars + 1):(NumVars + 2)], np.size(data[:, 0]), 1)
    y3 = np.reshape(data[:, (NumVars + 2):(NumVars + 3)], np.size(data[:, 0]), 1)
    y4 = np.reshape(data[:, (NumVars + 3):(NumVars + 4)], np.size(data[:, 0]), 1)

    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)

    stime = 0
    ftime = 0

    lr_model = LR()
    stime = time.time()
    lr_model.fit(X, y1)
    ftime = time.time()
    print("LR fit time:", (ftime-stime))
    stime = time.time()
    lr_model.predict([X[0]])
    ftime = time.time()
    print("LR predict time:", (ftime-stime))

    svr_model = SVR(kernel='linear', C=15, epsilon=0.2)
    stime = time.time()
    svr_model.fit(X, y1)
    ftime = time.time()
    print("SVR fit time:", (ftime-stime))
    stime = time.time()
    svr_model.predict([X[0]])
    ftime = time.time()
    print("SVR predict time:", (ftime-stime))

    grad_model = GradientBoostingRegressor(loss='lad', learning_rate=0.15, min_samples_split=2, n_estimators=140,
                                           max_depth=5, min_samples_leaf=2, max_features='sqrt', subsample=0.8,
                                           random_state=0)
    stime = time.time()
    grad_model.fit(X, y1)
    ftime = time.time()
    print("Grad Boost fit time:", (ftime-stime))
    stime = time.time()
    grad_model.predict([X[0]])
    ftime = time.time()
    print("Grad Boost predict time:", (ftime-stime))

    rf_model = RandomForestRegressor(n_estimators=37, max_depth=13, min_samples_split=2, min_samples_leaf=1,
                                     max_features=1, random_state=1)
    stime = time.time()
    rf_model.fit(X, y1)
    ftime = time.time()
    print("RF fit time:", (ftime - stime))
    stime = time.time()
    rf_model.predict([X[1]])
    ftime = time.time()
    print("RF predict time:", (ftime - stime))

    xg_model = xgb.XGBRegressor(learning_rate=0.1, n_estimators=100, max_depth=5, min_child_weight=2, gamma=1.9,
                                subsample=0.25, scale_pos_weight=1, colsample_bytree=0.7, verbosity=0, reg_alpha=0.2,
                                reg_lambda=0.1)
    stime = time.time()
    xg_model.fit(X, y1)
    ftime = time.time()
    print("XGB fit time:", (ftime - stime))
    stime = time.time()
    xg_model.predict([X[2]])
    ftime = time.time()
    print("XGB predict time:", (ftime - stime))

    mlp1_model = MLPRegressor(hidden_layer_sizes=3, learning_rate_init=0.12, random_state=1, batch_size=15,
                              max_iter=1000)
    stime = time.time()
    mlp1_model.fit(X, y1)
    ftime = time.time()
    print("MLP 1L fit time:", (ftime - stime))
    stime = time.time()
    mlp1_model.predict([X[3]])
    ftime = time.time()
    print("MLP 1L predict time:", (ftime - stime))

    mlp2_model = MLPRegressor(hidden_layer_sizes=(3, 6), learning_rate_init=0.12, random_state=1, batch_size=8,
                              max_iter=1000)
    stime = time.time()
    mlp2_model.fit(X, y1)
    ftime = time.time()
    print("MLP 2L fit time:", (ftime - stime))
    stime = time.time()
    mlp2_model.predict([X[4]])
    ftime = time.time()
    print("MLP 2L predict time:", (ftime - stime))


    '''
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

    # Grid Search ---------------------------------------------------------------------------------------------------
    '''
    # XGBoost
    max_dep = range(3, 11)
    min_c_w = range(1, 20)
    g_range = np.linspace(0, 2, 21)
    sub_range = np.linspace(0, 1, 21)
    col_range = np.linspace(0, 1, 21)
    alpha_range = np.linspace(0, 3, 31)
    lambda_range = np.linspace(0, 3, 31)
    l_range = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17,
               0.18, 0.19, 0.2, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.3]
    est_range = [50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200]
    param_grid = dict(max_depth=max_dep, min_child_weight=min_c_w)
    param_grid2 = dict(gamma=g_range)
    param_grid3 = dict(subsample=sub_range, colsample_bytree=col_range)
    param_grid4 = dict(reg_alpha=alpha_range, reg_lambda=lambda_range)
    param_grid5 = dict(learning_rate=l_range, n_estimators=est_range)
    # For sub_range/col_range: 0.25, 0.7 is ideal, for range 0.5-1, 0.55 and 0.5 are ideal...

    xg_reg = xgb.XGBRegressor(learning_rate=0.1, n_estimators=100, max_depth=3, min_child_weight=2, gamma=0,
                              subsample=0.8, scale_pos_weight=1, colsample_bytree=0.8, verbosity=0, reg_alpha=0,
                              reg_lambda=1)

    grid = GridSearchCV(xg_reg, param_grid5, cv=5, n_jobs=-1, scoring='neg_mean_squared_error')
    grid.fit(X, y4)

    print("Parameter used:", grid.best_params_)
    print("Best score:", grid.best_score_)
    '''

    '''
    # Gradient Boost
    num_e = range(20, 201, 10)
    max_dep = range(1, 11)
    num_sam_s = range(2, 5)
    min_sam_l = range(1, 5)
    max_feat = range(2, 10)
    sub_range = np.linspace(0.1, 1, 19)

    param_grid = dict(n_estimators=num_e)
    param_grid2 = dict(max_depth=max_dep, min_samples_split=num_sam_s, min_samples_leaf=min_sam_l)
    param_grid3 = dict(max_features=max_feat)
    param_grid4 = dict(subsample=sub_range)

    grad = GradientBoostingRegressor(loss='lad', learning_rate=0.15, min_samples_split=2, n_estimators=200, max_depth=3,
                                     min_samples_leaf=1, max_features='sqrt', subsample=0.8, random_state=0)

    grid = GridSearchCV(grad, param_grid4, cv=5, n_jobs=-1, scoring='neg_mean_squared_error')
    grid.fit(X, y4)

    print("Parameter used:", grid.best_params_)
    print("Best score:", grid.best_score_)
    '''

    '''
    # Random Forest
    feat_r = range(1, 10)
    dep_r = range(1, 19)
    split_r = range(2, 11)
    leaf_r = range(1, 11)
    est_r = range(20, 100)

    param_grid = dict(max_features=feat_r, max_depth=dep_r)
    param_grid2 = dict(min_samples_split=split_r, min_samples_leaf=leaf_r)
    param_grid3 = dict(n_estimators=est_r)

    rf = RandomForestRegressor(n_estimators=37, max_depth=14, min_samples_split=2, min_samples_leaf=1,
                               max_features=1, random_state=1)

    grid = GridSearchCV(rf, param_grid, cv=5, n_jobs=-1, scoring='neg_mean_squared_error')
    grid.fit(X, y4)

    print("Parameter used:", grid.best_params_)
    print("Best score:", grid.best_score_)
    '''

    '''
    # SVR - Linear
    C_r = [0.001, 0.01, 0.1, 1, 5, 10, 20, 50, 80, 100, 200, 300]
    E_r = [0.001, 0.01, 0.1, 1, 5, 10, 20, 50, 80, 100, 200, 300]
    C_r = range(10, 30)
    E_r = np.linspace(0.1, 0.7, 7)

    param_grid = dict(C=C_r, epsilon=E_r)

    svr = SVR(kernel='linear', C=27, epsilon=0.5)
    #svr = SVR(kernel='linear', degree=3, gamma='auto', C=100, epsilon=0.01)

    grid = GridSearchCV(svr, param_grid, cv=5, n_jobs=-1, scoring='neg_mean_squared_error')
    grid.fit(X, y4)

    print("Parameter used:", grid.best_params_)
    print("Best score:", grid.best_score_)
    '''

    '''
    # MLP
    layer_r = [(2,2), (2,3), (2,4), (2,5), (2,6),
               (3,2), (3,3), (3,4), (3,5), (3,6),
               (4,2), (4,3), (4,4), (4,5), (4,6),
               (5,2), (5,3), (5,4), (5,5), (5,6),
               (6,2), (6,3), (6,4), (6,5), (6,6)]
    # layer_r = range(2, 6)
    learn_r = np.linspace(0.01, 0.2, 20)
    batch_r = range(1, 25)

    param_grid = dict(hidden_layer_sizes=layer_r, batch_size=batch_r)
    param_grid2 = dict(learning_rate_init=learn_r)

    mlp = MLPRegressor(hidden_layer_sizes=(3, 5), learning_rate_init=0.12, random_state=1, batch_size=7, max_iter=1000)

    grid = GridSearchCV(mlp, param_grid2, cv=5, n_jobs=-1, scoring='neg_mean_squared_error')
    grid.fit(X, y4)

    print("Parameter used:", grid.best_params_)
    print("Best score:", grid.best_score_)
    '''

    '''
    # Feature Importance for Tree-Based Models (XGBoost, Gradient Boosting, Random Forest)
    rf = RandomForestRegressor(n_estimators=37, max_depth=13, min_samples_split=2, min_samples_leaf=1, max_features=1,
                               random_state=1)
    rf.fit(X, y1)
    print(rf.feature_importances_)
    '''

