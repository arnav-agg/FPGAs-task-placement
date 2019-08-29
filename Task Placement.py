import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression as LR
from sklearn.svm import SVR

import itertools
import random
import time

if __name__ == '__main__':
    # Data parsing
    data = 'HeatmapData - 9 Features.csv'
    data = np.array(np.genfromtxt(data, delimiter=','))
    NumVars = np.shape(data)[1] - 4

    X_full = np.reshape(data[:, 0:NumVars], (np.size(data[:, 0]), NumVars))
    y1_full = np.reshape(data[:, NumVars:(NumVars + 1)], np.size(data[:, 0]), 1)
    y2_full = np.reshape(data[:, (NumVars + 1):(NumVars + 2)], np.size(data[:, 0]), 1)
    y3_full = np.reshape(data[:, (NumVars + 2):(NumVars + 3)], np.size(data[:, 0]), 1)
    y4_full = np.reshape(data[:, (NumVars + 3):(NumVars + 4)], np.size(data[:, 0]), 1)

    # Creating a set of 100 random combinations of the possible task combinations...
    indices = range(65)
    possSets = list(itertools.combinations(indices, 4))
    # print(np.shape(possSets))
    random.seed(1)
    sets = random.sample(possSets, 1000)

    act_minPeak = []
    time_arr = []

    # Here is where the task placement begins --------------------------------------------------------------------------
    for sing_set in sets:
        test = list(reversed(sing_set))

        X = X_full
        y1 = y1_full
        y2 = y2_full
        y3 = y3_full
        y4 = y4_full

        X_t = X_full

        for task in test:
            X = np.delete(X, task, 0)
            y1 = np.delete(y1, task, 0)
            y2 = np.delete(y2, task, 0)
            y3 = np.delete(y3, task, 0)
            y4 = np.delete(y4, task, 0)

        placements = list(itertools.permutations(test))

        scaler = StandardScaler()
        scaler.fit(X)
        X = scaler.transform(X)
        X_t = scaler.transform(X_t)

        model_1 = SVR(kernel='linear', C=15, epsilon=0.2)
        model_2 = SVR(kernel='linear', C=27, epsilon=0.6)
        model_3 = SVR(kernel='linear', C=17, epsilon=0.001)
        model_4 = SVR(kernel='linear', C=18, epsilon=0.3)
        #model_1 = LR()
        #model_2 = LR()
        #model_3 = LR()
        #model_4 = LR()

        model_1.fit(X, y1)
        model_2.fit(X, y2)
        model_3.fit(X, y3)
        model_4.fit(X, y4)

        minPeakTemp = 200
        count = 0
        minIndice = 0

        stime = time.time()

        for task in placements:
            temp_array = []
            temp_array.append(model_1.predict([X_t[task[0]]]))
            temp_array.append(model_2.predict([X_t[task[1]]]))
            temp_array.append(model_3.predict([X_t[task[2]]]))
            temp_array.append(model_4.predict([X_t[task[3]]]))
            if max(temp_array) < minPeakTemp:
                minPeakTemp = max(temp_array)
                minIndice = count
            count = count + 1

        placement = placements[minIndice]

        ftime = time.time()

        temp_array = []
        temp_array.append(y1_full[placement[0]])
        temp_array.append(y2_full[placement[1]])
        temp_array.append(y3_full[placement[2]])
        temp_array.append(y4_full[placement[3]])

        act_minPeak.append(max(temp_array))

        time_arr.append(ftime-stime)

    print("Average time:", sum(time_arr)/len(time_arr))
    print("Minimum peak temp from prediction:", sum(act_minPeak)/len(act_minPeak))


