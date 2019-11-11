import numpy as np
import itertools
import random
import sys

if __name__ == '__main__':
    N = 8
    num_samples = int(sys.argv[1])
    # Data parsing
    data = '86data_8node.csv'
    data = np.array(np.genfromtxt(data, delimiter=','))
    NumVars = np.shape(data)[1] - N

    X_full = np.reshape(data[:, 0:NumVars], (np.size(data[:, 0]), NumVars))
    y_full = []
    for i in range(N):
        y_full.append(np.reshape(data[: , NumVars + i : NumVars + i + 1], np.size(data[:, 0]), 1))

    # Creating a set of 100 random combinations of the possible task combinations...
    indices = range(86)
    #possSets = list(itertools.combinations(indices, N))
    # print(np.shape(possSets))
    random.seed(1)
    #sets = random.sample(possSets, 20)

    minPeakArr = []
    maxPeakArr = []
    avePeakArr = []

    for i in range(num_samples):
        workload = random.sample(indices, N)
        placements = list(itertools.permutations(workload))

        minPeakTemp = 200
        maxPeakTemp = 0
        count = 0
        minIndice = 0
        maxIndice = 0

        oblTempArr = []

        for task in placements:
            temp_array = []
            for i in range(N):
                temp_array.append(y_full[i][task[i]])

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



