import numpy as np
import itertools
import random

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

    test = sets[0]
    test = list(reversed(test))

    minPeakArr = []
    maxPeakArr = []
    avePeakArr = []

    for test in sets:
        placements = list(itertools.permutations(test))

        minPeakTemp = 200
        maxPeakTemp = 0
        count = 0
        minIndice = 0
        maxIndice = 0

        oblTempArr = []

        for task in placements:
            temp_array = []
            temp_array.append(y1_full[task[0]])
            temp_array.append(y2_full[task[1]])
            temp_array.append(y3_full[task[2]])
            temp_array.append(y4_full[task[3]])

            if max(temp_array) < minPeakTemp:
                minPeakTemp = max(temp_array)
                minIndice = count
            if max(temp_array) > maxPeakTemp:
                maxPeakTemp = max(temp_array)
                maxIndice = count
            count = count + 1

            oblTempArr.append(max(temp_array))

        avePeakTemp = sum(oblTempArr)/len(oblTempArr)

        minPeakArr.append(minPeakTemp)
        maxPeakArr.append(maxPeakTemp)
        avePeakArr.append(avePeakTemp)
        '''
        print(minPeakTemp)
        print(maxPeakTemp)
        print(avePeakTemp)
        print(placements[minIndice])
        print(placements[maxIndice])
        '''

    print("")
    print("Ave minimum peak temp:", sum(minPeakArr)/len(minPeakArr))
    print("Ave maximum peak temp:", sum(maxPeakArr)/len(maxPeakArr))
    print("Ave oblivious peak temp:", sum(avePeakArr)/len(avePeakArr))



