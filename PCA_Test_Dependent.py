import itertools
import numpy as np
import pandas as pd
import random as rand
from sklearn.linear_model import LinearRegression as LR
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler
import time

from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt

indices = range(10)
numFeatures = 55


def locate_min(a):
    smallest = min(a)
    return smallest, [index for index, element in enumerate(a) if smallest == element]


def locate_max(a):
    largest = max(a)
    return largest, [index for index, element in enumerate(a) if largest == element]


def parseData(appUsage, physical1, physical2, physical3, physical4):
    usage = np.array(np.genfromtxt(appUsage, delimiter=','))
    p = []
    p.append(np.array(np.genfromtxt(physical1, delimiter=',')))
    p.append(np.array(np.genfromtxt(physical2, delimiter=',')))
    p.append(np.array(np.genfromtxt(physical3, delimiter=',')))
    p.append(np.array(np.genfromtxt(physical4, delimiter=',')))
    # X = np.reshape(data[:,0:2],(np.size(data[:,0]), 2))
    # y = np.reshape(data[:,2],(np.size(data[:,1]),1))
    return usage, p


def placement(usage, phy):
    # All possible sets of 4 selected from the 10 benchmarks. Pick 4 out of 10 is 210.
    possSets = list(itertools.combinations(indices, 4))
    total = len(possSets)
    print("# of possible sets:", total)
    print("")

    best_temp = []
    best_pwr = []
    ob_temp = []
    ob_pwr = []
    my_temp = []
    my_pwr = []
    mse = []
    rand.seed(1)

    for run in range(1):
        # 56 of the sets are chosen to train and (210 - 168 = 42) are used for testing. 80-20 split.
        # train_sets = list(rand.sample(possSets, 56))
        train_sets = list(rand.sample(possSets, 168))
        test_sets = list(set(possSets) - set(train_sets))
        train_data = []
        error = []

        # print(len(train_sets))
        # print(len(test_sets))

        # Total of (#_train_sets * 24), 4032 in this case where #_train_sets = 168. 24 is the number of permutations
        for comb in train_sets:
            # For each of the X different combinations of 4 in train_sets, use every permutation (4! = 24) of them so
            # each node indicated by the indice of the placements array ([0][1][2][3]) will get each benchmark.
            placements = list(itertools.permutations(comb))

            for p in placements:
                entry = []
                peakTemp = 0
                sumPow = 0
                for i in p:
                    for indice in range(numFeatures):
                        entry.append(usage[i][indice])
                # For the first benchmark, get the first node's power/temperature, for the second get the second node...
                for m in range(4):
                    temp = phy[m][p[m]][0]
                    if peakTemp < temp:
                        peakTemp = temp
                    sumPow = sumPow + phy[m][p[m]][1]
                # Append the peak temperature and the sum of the power onto the entry, then append entire thing onto the
                # big train_data array.
                entry.append(peakTemp)
                entry.append(sumPow)

                train_data.append(entry)

        train_data = np.array(train_data)

        test_comb = []
        # For each testing set, gather all of the temperature/power data for the benchmarks and node combinations
        # Each "comb" set is a set of 24. Total of #_test_sets * 24 different values.
        for comb in test_sets:
            placements = list(itertools.permutations(comb))
            for p in placements:
                entry = []
                peakTemp = 0
                sumPow = 0
                for i in p:
                    for indice in range(numFeatures):
                        entry.append(usage[i][indice])
                for m in range(4):
                    temp = phy[m][p[m]][0]
                    if peakTemp < temp:
                        peakTemp = temp
                    sumPow = sumPow + phy[m][p[m]][1]
                entry.append(peakTemp)
                entry.append(sumPow)

                test_comb.append(entry)

        test_comb = np.array(test_comb)
        print("Test_comb shape:", np.shape(test_comb))

        X_train, y_train, X_test, y_test = normAllData(train_data, test_comb)

        # PCA stuff begins here ---------------------------------------------------------------------------------------
        pca = PCA(0.95)
        pca.fit(X_train)
        '''
        sns.heatmap(abs(pca.components_))
        plt.show()
        '''
        # -------------------------------------------------------------------------------------------------------------
        compTotal = []
        for comp in range(pca.n_components_):
            compTotal.append(sum(abs(pca.components_[comp])))
        print(compTotal)

        finalArray = []
        for i in range(numFeatures):
            value = 0
            for comp in range(pca.n_components_):
                weightVal = 0
                for j in range(4):
                    weightVal = weightVal + abs(pca.components_[comp][i + j*numFeatures])
                weightVal = weightVal/compTotal[comp]
                value = value + (pca.explained_variance_ratio_[comp] * weightVal)
            finalArray.append(value)

        print("Length:", len(finalArray))
        print("Maximum:", max(finalArray), finalArray.index(max(finalArray)))
        print("Minimum:", min(finalArray), finalArray.index(min(finalArray)))

        print("")
        print("Number of principal components:", pca.n_components_)
        print("Component 0 variance ratios:")
        print(pca.explained_variance_ratio_)
        print(pca.explained_variance_ratio_[0])
        print("Total variance accounted for:", sum(pca.explained_variance_ratio_))
        print("Component 0 weighting:")
        print(abs(pca.components_[0][0]) + abs(pca.components_[0][numFeatures]) + abs(pca.components_[0][numFeatures*2]) + abs(pca.components_[0][numFeatures*3]))
        print("")
        # -------------------------------------------------------------------------------------------------------------

        X_train = pca.transform(X_train)
        X_test = pca.transform(X_test)

        # print("start train: %.20f" % time.time())
        lr = fitLR(X_train, y_train)

        '''
        print("LR Coefficients:")
        print(lr.coef_)
        print("LR intercept:", lr.intercept_)
        print("")
        '''

        # print("finish train: %.20f" % time.time())

        '''
        best_placement_temp = []
        ob_placement_temp = []
        my_placement_temp = []

        best_placement_pwr = []
        ob_placement_pwr = []
        my_placement_pwr = []

        stime = 0
        ftime = 0

        stime = stime + time.time()
        # Predict the value for each of the 24 permutations in this combination.
        y_pred = lr.predict(X_test)
        ftime = ftime + time.time()
        '''

        '''
            error.append((y_pred - y_test))

            # best placement
            # Lowest temperature recorded from the data
            smallest, best_min_indices = locate_min(y_test)  # might have multiple min
            best_placement_temp.append(smallest)
            tmp_min_pwr = 200
            # If there are multiple minimum temperatures, use the lowest power
            for index in best_min_indices:
                if test_comb[index, (numFeatures*4 + 1)] < tmp_min_pwr:
                    tmp_min_pwr = test_comb[index, (numFeatures*4 + 1)]

            best_placement_pwr.append(tmp_min_pwr)

            # oblivious placement
            # Average of all the temperatures/powers from the data
            ob_placement_temp.append(np.mean(y_test))
            ob_placement_pwr.append(np.mean(test_comb[:, (numFeatures*4 + 1)]))

            # my placement
            # Minimum value from the models prediction. Others values are from the data
            pred_min_index = np.argmin(y_pred)
            my_placement_temp.append(y_test[pred_min_index])
            if y_test[pred_min_index] != smallest:
                print(comb)
            my_placement_pwr.append(test_comb[pred_min_index, (numFeatures*4 + 1)])

        stime = stime / len(test_sets)
        ftime = ftime / len(test_sets)
        duration = ftime - stime
        # print("predict: %.20f" % duration)
        # print("finish predict: %.20f" % ftime)
        # best
        best_temp.append(np.mean(best_placement_temp))
        my_ob_diff = np.array(ob_placement_temp) - np.array(my_placement_temp)
        max_diff, max_diff_indices = locate_max(my_ob_diff)
        # print('number of max is ', len(max_diff_indices))
        # print('max_diff is', max_diff, 'combination is ', test_sets[max_diff_indices[0]])

        min_diff, min_diff_indices = locate_min(my_ob_diff)
        # print('number of min is ', len(min_diff_indices))
        # print('min_diff is', min_diff, 'combination is ', test_sets[min_diff_indices[0]])

        best_pwr.append(np.mean(best_placement_pwr))

        # oblivious
        ob_temp.append(np.mean(ob_placement_temp))
        ob_pwr.append(np.mean(ob_placement_pwr))
        # print(np.max(ob_placement_pwr))
        # print(ob_placement_pwr)
        diff = np.array(ob_placement_pwr) - np.array(best_placement_pwr)
        pwr_idx = np.argmax(diff)
        # print('best ', float(diff[pwr_idx])/ob_placement_pwr[pwr_idx])
        # print('best ', diff[pwr_idx])

        # mine
        my_temp.append(np.mean(my_placement_temp))
        my_pwr.append(np.mean(my_placement_pwr))
        # print(my_placement_pwr[np.argmax(ob_placement_pwr)])
        # print(my_placement_pwr)
        diff = np.array(ob_placement_pwr) - np.array(my_placement_pwr)
        pwr_idx = np.argmax(diff)
        print(float(diff[pwr_idx]) / ob_placement_pwr[pwr_idx])
        print(diff[pwr_idx])
        print(ob_placement_pwr[pwr_idx])

        error = np.array(error)
        tmp = error.flatten()
    # print(np.mean(np.abs(tmp)))

    # mse.append(error.flatten())

    print(np.mean(best_temp), np.mean(best_pwr))
    print(np.mean(ob_temp), np.mean(ob_pwr))
    print(np.mean(my_temp), np.mean(my_pwr))
    mse = np.array(mse)
    # print(ob_pwr)
    # print(my_pwr)

    # print(np.mean(mse.flatten()**2))
    '''

    print('finish')


def normAllData(train_data, test_data):
    X_train = np.reshape(train_data[:, 0:(numFeatures*4)], (np.size(train_data[:, 0]), (numFeatures*4)))
    X_test = np.reshape(test_data[:, 0:(numFeatures*4)], (np.size(test_data[:, 0]), (numFeatures*4)))
    scaler = StandardScaler()
    # Only fit the training data
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    y_train = np.reshape(train_data[:, (numFeatures*4):(numFeatures*4 + 1)], (np.size(train_data[:, 0]), 1))
    y_test = np.reshape(test_data[:, (numFeatures*4):(numFeatures*4 + 1)], (np.size(test_data[:, 0]), 1))
    y_train = np.ravel(y_train)
    y_test = np.ravel(y_test)
    return X_train, y_train, X_test, y_test


def normDataset(data):
    # print("Shape:", np.shape(data))
    # print("Data[1]:", data[1])
    X = np.reshape(data[:, 0:(numFeatures*4)], (np.size(data[:, 0]), (numFeatures*4)))
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)

    y = np.reshape(data[:, (numFeatures*4):(numFeatures*4 + 1)], (np.size(data[:, 0]), 1))
    y = np.ravel(y)
    return X, y


def fitLR(X, y):
    lr = LR()
    lr.fit(X, y)

    # For some error checking
    lr2 = LR()
    y_pred = cross_val_predict(lr, X, y, cv=10)
    mse = np.mean((y_pred - y) ** 2)
    abse = np.mean(np.abs(y_pred - y))
    print("LR: ", mse)
    print("LR: ", abse)
    print("\n")

    return lr


def fitMLP(X, y):
    mlp = MLPRegressor(hidden_layer_sizes=(5,), learning_rate_init=0.1, random_state=0,
                       batch_size=100)  # , learning_rate='adaptive')
    mlp.fit(X, y)
    return mlp


def fitNN(X, y):
    nn = KNeighborsRegressor(n_neighbors=1, p=2)
    # y = np.ravel(y)
    y_pred = cross_val_predict(nn, X, y, cv=10)
    mse = np.mean((y_pred - y) ** 2)
    abse = np.mean(np.abs(y_pred - y))
    print("NN: ", mse)
    print("NN: ", abse)
    print("\n")


def fitRF(X, y):
    rf = RandomForestRegressor(n_estimators=120, max_features='auto', max_depth=6, random_state=1)
    rf.fit(X, y)
    return rf


if __name__ == '__main__':
    appU = 'ResourceUsageUpd.csv'
    p1 = "p1.csv"
    p2 = "p2.csv"
    p3 = "p3.csv"
    p4 = "p4.csv"
    usage, p = parseData(appU, p1, p2, p3, p4)
    placement(usage, p)

