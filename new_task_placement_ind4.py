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
num_tasks = 86;

# SA
# rejects = 0;

# svr model parameters
svr_params = [('linear', 27, 0.5),
              ('linear', 27, 0.5),
              ('linear', 27, 0.5),
              ('linear', 27, 0.5)]
# svr_params = [('linear', 15, 0.2), \
#              ('linear', 27, 0.6), \
#              ('linear', 17, 0.001), \
#              ('linear', 18, 0.3)]

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


def brute_force2(workload, y_full, pow_full):
    N = len(workload)
    placements = list(itertools.permutations(workload))
    minPeakTemp = 500
    count = 0
    minIndice = 0
    for task in placements:
        temp_array = []
        for i in range(N):
            temp_array.append(y_full[i][task[i]])
        if max(temp_array) < minPeakTemp:
            minPeakTemp = max(temp_array)
            minIndice = count
        count = count + 1

    maxPeakTemp = 0
    count = 0
    maxIndice = 0
    for task in placements:
        temp_array = []
        for i in range(N):
            temp_array.append(y_full[i][task[i]])
        if max(temp_array) > maxPeakTemp:
            maxPeakTemp = max(temp_array)
            maxIndice = count
        count = count + 1

    obl_Peak_array = []
    obl_Pow_array = []
    for task in placements:
        temp_array = []
        pow_array = []
        for i in range(N):
            temp_array.append(y_full[i][task[i]])
            pow_array.append(pow_full[i][task[i]])
        obl_Peak_array.append(max(temp_array))
        obl_Pow_array.append(max(pow_array))

    obl_temperature = sum(obl_Peak_array) / len(obl_Peak_array)
    obl_power = sum(obl_Pow_array) / len(obl_Pow_array)

    real_best_p = placements[minIndice]
    real_worst_p = placements[maxIndice]
    return real_best_p, real_worst_p, obl_temperature, obl_power


def simulated_annealing(workload, model, X):
    # initial placement
    placement = [t for t in workload]
    # initial temperature
    T = 1000
    # temperature reduction rate
    r = 0.85
    # number of trails at each temperature
    M = 4
    steps = 0
    time_taken = 0
    # time in seconds since the epoch
    stime = time.time()
    while time_taken < 1 and steps < 4:
        placement, rejects = perturb(T, M, placement, model, X)
        # print "rejects: ", rejects
        if rejects >= 0.9 * M:
            break
        steps += 1
        # print "temp", T
        # T = pow(r, steps) * T
        T = r * T
        time_taken = time.time() - stime
    #print(placement)
    return placement


def perturb(T, M, placement, model, X):
    p = [t for t in placement]
    best_p = [t for t in placement]
    # print "best start", len(best_p), best_p, placement
    cost = get_cost(p, model, X)
    min_cost = cost
    uphills = 0
    rejects = 0
    Mt = 0
    while Mt < M and uphills < M / 2:
        move = random.randint(0, 1)
        #move = random.randint(0, 1)
        if move == 0:
            neighbor_p = swap(p)
        elif move == 1:
            neighbor_p = reverse(p)
        else:  # move == 2
            neighbor_p = translate(p)
        if neighbor_p == None:
            continue
        # print move, p, neighbor_p
        new_cost = get_cost(neighbor_p, model, X)
        delta = new_cost - cost
        probability = random.uniform(0, 1)
        if delta <= 0 or probability < math.exp(-delta / T):
            p = [t for t in neighbor_p]
            cost = new_cost
            if delta > 0:
                uphills += 1
            if new_cost < min_cost:
                best_p = [t for t in neighbor_p]
                min_cost = new_cost
                # print min_cost
        else:
            rejects += 1
        # print "rejects: ", rejects
        Mt += 1
    # print "best", len(best_p), best_p, min_cost, placement
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
        # print "in swap while"
        j = random.randint(0, N - 1)
    w_c = [t for t in workload]
    #print len(w_c), i
    t1 = w_c[i]
    w_c[i] = w_c[j]
    w_c[j] = t1
    # print("swap", len(workload))
    return w_c


def reverse(workload):
    N = len(workload)
    # random number between 0 and N-3 (both sides are inclusive)
    start = random.randint(0, N - 3)
    # random number between start+2 and N-1 (both sides are inclusive)
    end = random.randint(start + 2, N - 1)
    # length = randint(1, N - start + 1)
    # get sub-list from start to end (inclusive)
    sub_workload = workload[start: end + 1]
    sub_workload.reverse()
    if end == N - 1:
        new_seq = workload[0: start] + sub_workload
    else:
        new_seq = workload[0: start] + sub_workload + workload[end + 1:]
    # print("reverse", len(new_seq), len(workload))
    return new_seq


def translate(workload):
    N = len(workload)
    # random number between 0 and N-2 (both sides are inclusive)
    start = random.randint(0, N - 2)
    # random number between start+1 and N-1 (both sides are inclusive)
    end = random.randint(start + 1, N - 1)
    # get sub-list from start to end (inclusive)
    sub_workload = workload[start: end + 1]

    workload_cpy = [t for t in workload]
    del workload_cpy[start: end + 1]
    left_len = len(workload_cpy)
    if left_len < 1:
        return None

    insert_idx = random.randint(0, left_len - 1)
    while insert_idx == start and insert_idx != 0:
        # print("in translate while", insert_idx)
        insert_idx = random.randint(0, left_len - 1)
    new_seq = workload_cpy[0: insert_idx] + sub_workload + workload_cpy[insert_idx:]
    # print("translate", len(new_seq), len(workload))
    # if new_seq == workload_cpy:

    return new_seq


if __name__ == '__main__':
    # set this argument to 1 to use the brute_force method
    # use_brute_force = 0
    random.seed(0)
    use_brute_force = int(sys.argv[1])
    num_samples = int(sys.argv[2])
    #use_brute_force = 0
    #num_samples = 6
    # number of tasks to place
    N = 4
    # initialize ML models
    model = []
    for i in range(N):
        p = svr_params[i]
        model.append(SVR(kernel=p[0], C=p[1], epsilon=p[2]))
        # model.append(LR())

    # Data parsing
    data = '86data.csv'
    data = np.array(np.genfromtxt(data, delimiter=','))
    print data.shape
    NumVars = np.shape(data)[1] - 2*N
    X_full = np.reshape(data[:, 0: NumVars], (np.size(data[:, 0]), NumVars))

    y_full = []
    for i in range(N):
        y_full.append(np.reshape(data[:, NumVars + i: NumVars + i + 1], np.size(data[:, 0]), 1))

    pow_full = []
    for i in range(N):
        pow_full.append(np.reshape(data[:, NumVars + N + i: NumVars + N + i + 1], np.size(data[:, 0]), 1))

    indices = range(num_tasks)
    workloads = []
    for i in range(num_samples):
        workloads.append(random.sample(indices, N))

    # If N is large, e.g., 8, it takes lots of memory to get all posssible sets
    # poss_sets = list(itertools.combinations(indices, N))
    # Creating a set of 100 random combinations of the possible task combinations...
    # sets = random.sample(poss_sets, 1)

    real_minPeak = []
    real_minPow = []
    worst_Peak = []
    worst_Pow = []
    obl_Peak = []
    obl_Pow = []
    act_minPeak = []
    act_minPow = []
    time_arr = []
    runs = 0
    
    # Here is where the task placement begins --------------------------------------------------------------------------
    for workload in workloads:
        train_indices = list(set(indices) - set(workload))
        X = X_full[train_indices, :]
        y = []
        for i in range(N):
            y.append(y_full[i][train_indices])

        X_t = X_full[:]
        scaler = StandardScaler()
        scaler.fit(X)
        X = scaler.transform(X)
        X_t = scaler.transform(X_t)
        # train temperature models for each node
        for i in range(N):
            model[i].fit(X, y[i])

        w_c = [t for t in workload]
        stime = time.time()
        if use_brute_force == 1:
            # print("using brute force")
            placement = brute_force(w_c, model, X_t)
        else:
            placement = simulated_annealing(w_c, model, X_t)

        ftime = time.time()

        # Shouldn't need to change for simulated_annealing
        real_p, real_worst, obl_temp, obl_power = brute_force2(workload, y_full, pow_full)

        obl_Peak.append(obl_temp)
        obl_Pow.append(obl_power)

        worst_Peak_array = []
        worst_Pow_array = []
        for i in range(N):
            worst_Peak_array.append(y_full[i][real_worst[i]])
            worst_Pow_array.append(pow_full[i][real_worst[i]])

        worst_Peak.append(max(worst_Peak_array))
        worst_Pow.append(max(worst_Pow_array))

        temp_array_real = []
        pow_array_real = []
        for i in range(N):
            temp_array_real.append(y_full[i][real_p[i]])
            pow_array_real.append(pow_full[i][real_p[i]])

        real_minPeak.append(max(temp_array_real))
        real_minPow.append(max(pow_array_real))

        temp_array = []
        pow_array = []
        for i in range(N):
            temp_array.append(y_full[i][placement[i]])
            pow_array.append(pow_full[i][placement[i]])

        act_minPeak.append(max(temp_array))
        act_minPow.append(max(pow_array))

        time_arr.append(ftime - stime)
        runs += 1
        #if runs % 5 == 0:
            #print "finished ", runs, "runs"

    print("Average time:", sum(time_arr) / len(time_arr))
    print("Thermal-aware peak temp from prediction:", sum(act_minPeak) / len(act_minPeak))
    print("Thermal-aware peak power from prediction:", sum(act_minPow) / len(act_minPow))
    print("Optimal minimum peak temp:", sum(real_minPeak) / len(real_minPeak))
    print("Optimal minimum peak power:", sum(real_minPow) / len(real_minPow))
    print("Worst peak temp on average:", sum(worst_Peak) / len(worst_Peak))
    print("Worst peak power on average:", sum(worst_Pow) / len(worst_Pow))
    print("Worst of worst peak temp:", max(worst_Peak) )
    print("Worst of worst peak power:", max(worst_Pow) )
    print("Oblivious temp:", sum(obl_Peak) / len(obl_Peak))
    print("Oblivious power:", sum(obl_Pow) / len(obl_Pow))

