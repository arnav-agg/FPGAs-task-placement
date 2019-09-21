import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression as LR
from sklearn.svm import SVR

import itertools
import random
import time
import sys
import math

# total number of tasks
num_tasks = 65;

# SA 
#rejects = 0;

# svr model parameters
#svr_params = [('linear', 15, 0.2), \
#              ('linear', 27, 0.6), \
#              ('linear', 17, 0.001), \
#              ('linear', 18, 0.3)]
svr_params = [('linear', 15, 0.2), \
              ('linear', 27, 0.6), \
              ('linear', 17, 0.001), \
              ('linear', 18, 0.3), \
              ('linear', 15, 0.2), \
              ('linear', 27, 0.6), \
              ('linear', 17, 0.001), \
              ('linear', 18, 0.3)]

def brute_force(workload, model, X):
    N = len(workload)
    placements = list(itertools.permutations(workload))
    minPeakTemp = 500
    count = 0
    minIndice = 0
    for task in placements:
        temp_array = []
        for i in range(N):
            temp_array.append(model[i].predict([X[task[i]]]))
        if max(temp_array) < minPeakTemp:
            minPeakTemp = max(temp_array)
            minIndice = count
        count = count + 1

    best_p = placements[minIndice]
    return best_p

def simulated_annealing(workload, model, X):
    # initial placement
    placement = workload
    # initial temperature
    T = 1000
    # temperature reduction rate
    r = 0.85
    # number of trails at each temperature
    M = 10
    steps = 0
    time_taken = 0
    # time in seconds since the epoch
    stime = time.time()
    while time_taken < 3 and steps < 10:
        placement, rejects = perturb(T, M, placement, model, X)
        #print "rejects: ", rejects
        if rejects >= 0.9 * M:
            break
        steps += 1;
        #print "temp", T
        #T = pow(r, steps) * T
        T = r * T
        time_taken = time.time() - stime
    print placement
    return placement


def perturb(T, M, placement, model, X):
    p = placement[:]
    best_p = placement[:]
    #print "best start", len(best_p), best_p, placement
    cost = get_cost(p, model, X)
    min_cost = cost
    uphills = 0
    rejects = 0
    Mt = 0
    while Mt < M and uphills < M/2:
        move = random.randint(0, 2)
        if move == 0:
            neighbor_p = swap(p)
        elif move == 1:
            neighbor_p = reverse(p)
        else: # move == 2
            neighbor_p = translate(p)
        if neighbor_p == None:
            continue
        #print move, p, neighbor_p
        new_cost = get_cost(neighbor_p, model, X)
        delta = new_cost - cost
        probability = random.uniform(0, 1)
        if delta <= 0 or probability < math.exp(-delta/T):
            p = neighbor_p[:]
            cost = new_cost
            if delta > 0:
                uphills += 1
            if new_cost < min_cost:
                best_p = neighbor_p[:]
                min_cost = new_cost
                #print min_cost
        else:
            rejects += 1
        #print "rejects: ", rejects
        Mt += 1
    #print "best", len(best_p), best_p, min_cost, placement
    return best_p, rejects

def get_cost(placement, model, X):
    N = len(placement)
    cost = 0
    Ntemps = []
    for i in range(N):
        Ntemps.append(model[i].predict([X[placement[i]]]))
    # cost is the system peak temperature
    cost = max(Ntemps)
    return cost

def swap(workload):
    N = len(workload)
    i = random.randint(0, N - 1)
    j = random.randint(0, N - 1)
    while j == i:
        #print "in swap while"
        j = random.randint(0, N - 1)

    t1 = workload[i]
    workload[i] = workload[j]
    workload[j] = t1
    #print "swap", len(workload)
    return workload

def reverse(workload):
    N = len(workload)
    # random number between 0 and N-3 (both sides are inclusive)
    start = random.randint(0, N - 3)
    # random number between start+2 and N-1 (both sides are inclusive)
    end = random.randint(start + 2, N - 1)
    #length = randint(1, N - start + 1)
    # get sub-list from start to end (inclusive)
    sub_workload = workload[start : end + 1]
    sub_workload.reverse()
    if end == N-1:
        new_seq = workload[0 : start] + sub_workload
    else:
        new_seq = workload[0 : start] + sub_workload + workload[end + 1 : ]
    #print "reverse", len(new_seq), len(workload)
    return new_seq

def translate(workload):
    N = len(workload)
    # random number between 0 and N-2 (both sides are inclusive)
    start = random.randint(0, N - 2)
    # random number between start+1 and N-1 (both sides are inclusive)
    end = random.randint(start + 1, N - 1)
    # get sub-list from start to end (inclusive)
    sub_workload = workload[start : end + 1]

    workload_cpy = workload[:]
    del workload_cpy[start : end + 1]
    left_len = len(workload_cpy)
    if left_len < 1:
        return None

    insert_idx = random.randint(0, left_len - 1)
    while insert_idx == start and insert_idx != 0:
        #print "in translate while", insert_idx
        insert_idx = random.randint(0, left_len - 1)
    new_seq = workload_cpy[0 : insert_idx] + sub_workload + workload_cpy[insert_idx : ]
    #print "translate", len(new_seq), len(workload)
    #if new_seq == workload_cpy:

    return new_seq

if __name__ == '__main__':
    # set this argument to 1 to use the brute_force method
    #use_brute_force = 0
    use_brute_force = int(sys.argv[1])
    num_samples = int(sys.argv[2])
    # number of tasks to place
    N = 8;
    # initialize ML models
    model = []
    for i in range(N):
        p = svr_params[i]
        model.append(SVR(kernel=p[0], C=p[1], epsilon=p[2]))
        #model.append(LR())

    # Data parsing
    data = 'data.csv'
    data = np.array(np.genfromtxt(data, delimiter=','))
    NumVars = np.shape(data)[1] - N
    X_full = np.reshape(data[:, 0 : NumVars], (np.size(data[:, 0]), NumVars))

    y_full = []
    for i in range(N):
        y_full.append(np.reshape(data[: , NumVars + i : NumVars + i + 1], np.size(data[:, 0]), 1))

    indices = range(num_tasks)
    # If N is large, e.g., 8, it takes lots of memory to get all posssible sets
    #poss_sets = list(itertools.combinations(indices, N))
    # Creating a set of 100 random combinations of the possible task combinations...
    random.seed(1)
    #sets = random.sample(poss_sets, 1)

    act_minPeak = []
    time_arr = []
    runs = 0

    # Here is where the task placement begins --------------------------------------------------------------------------
    for i in range(num_samples):
        #workload = list(single_set)
        workload = random.sample(indices, N)

        #train_indices = list(set(indices) - set(single_set))
        train_indices = list(set(indices) - set(workload))
        X = X_full[train_indices, :]
        y = []
        for i in range(N):
            y.append(y_full[i][train_indices])

        X_t = X_full
        scaler = StandardScaler()
        scaler.fit(X)
        X = scaler.transform(X)
        X_t = scaler.transform(X_t)
        # train temperature models for each node
        for i in range(N):
            model[i].fit(X, y[i])

        stime = time.time()
        if use_brute_force == 1:
            print "using brute force"
            placement = brute_force(workload, model, X_t)
        else:
            placement = simulated_annealing(workload, model, X_t)
        
        ftime = time.time()

        temp_array = []
        for i in range(N):
            temp_array.append(y_full[i][placement[i]])

        act_minPeak.append(max(temp_array))

        time_arr.append(ftime-stime)
        runs += 1
        if runs % 5 == 0:
            print "finished ",runs, "runs"

    print("Average time:", sum(time_arr)/len(time_arr))
    print("Minimum peak temp from prediction:", sum(act_minPeak)/len(act_minPeak))
    
