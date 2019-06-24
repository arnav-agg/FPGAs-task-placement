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

indices = range(10)

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
	#X = np.reshape(data[:,0:2],(np.size(data[:,0]), 2))
	#y = np.reshape(data[:,2],(np.size(data[:,1]),1))
	return usage, p

def placement(usage, phy):
	possSets = list(itertools.combinations(indices, 4))
	total = len(possSets)
	print(total)

	best_temp = []
	best_pwr = []
	ob_temp = []
	ob_pwr = []
	my_temp = []
	my_pwr = []
	data = []
	rand.seed(1)
	all_sets = list(itertools.combinations(indices, 4)) #[(2, 3, 6, 7)]
	#all_sets = [(1, 4, 7, 9)]
	err = []

	for run in all_sets:
		#useIndices = list(rand.sample(indices, 6))
		#leftIndices = list(set(indices) - set(useIndices))
		#train_sets = list(itertools.combinations(useIndices, 4))
		#test_sets = list(itertools.combinations(leftIndices, 4))

		test_sets = list(itertools.combinations(run, 4)) 
		leftIndices = list(set(indices) - set(run))
		train_sets = list(itertools.combinations(leftIndices, 4))
		#print(len(train_sets))
		#print(len(test_sets))
		train_data = []

		for comb in train_sets:
			placements = list(itertools.permutations(comb))
			#print(len(placements))
			for p in placements:
				entry = []
				peakTemp = 0
				sumPow = 0
				for i in p:
					entry.append(usage[i][0])
					entry.append(usage[i][1])
					entry.append(usage[i][2])
				for m in range(4):
					temp = phy[m][p[m]][0]
					if peakTemp < temp:
						peakTemp = temp
					sumPow = sumPow + phy[m][p[m]][1]
				entry.append(peakTemp)
				entry.append(sumPow)

				train_data.append(entry)

		train_data = np.array(train_data)
		X_train, y_train = normDataset(train_data)
		lr = fitRF(X_train, y_train)

		best_placement_temp = []
		worst_placement_temp = []
		ob_placement_temp = []
		my_placement_temp = []

		best_placement_pwr = []
		worst_placement_pwr = []
		ob_placement_pwr = []
		my_placement_pwr = []

		for comb in test_sets:
			data_line = []
			for i in range(4):
				data_line.append(comb[i])
			test_comb = []
			placements = list(itertools.permutations(comb))
			for p in placements:
				entry = []
				peakTemp = 0
				sumPow = 0
				for i in p:
					entry.append(usage[i][0])
					entry.append(usage[i][1])
					entry.append(usage[i][2])
				for m in range(4):
					temp = phy[m][p[m]][0]
					if peakTemp < temp:
						peakTemp = temp
					sumPow = sumPow + phy[m][p[m]][1]
				entry.append(peakTemp)
				entry.append(sumPow)

				test_comb.append(entry)
			test_comb = np.array(test_comb)
			X_test, y_test = normDataset(test_comb)
			y_pred = lr.predict(X_test)
			#print(y_test)
			#print(y_pred)
			#err = np.sqrt(np.mean((y_test - y_pred)**2))
			#print(err)
			err.append((y_test - y_pred))

			# best/worse placement
			smallest, best_min_indices = locate_min(y_test)  # might have multiple min 
			best_placement_temp.append(smallest)
			largest, worst_max_indices = locate_max(y_test) 
			worst_placement_temp.append(largest)
			tmp_min_pwr = 200
			for index in best_min_indices:
				if test_comb[index, 13] < tmp_min_pwr:
					tmp_min_pwr = test_comb[index, 13]
			tmp_max_pwr = 0
			for index in worst_max_indices:
				if test_comb[index, 13] > tmp_max_pwr:
					tmp_max_pwr = test_comb[index, 13]

			best_placement_pwr.append(tmp_min_pwr)
			worst_placement_pwr.append(tmp_max_pwr)

			data_line.append(smallest)
			data_line.append(largest)
			# oblivious placement
			ob_placement_temp.append(np.mean(y_test))
			ob_placement_pwr.append(np.mean(test_comb[:,13]))

			# my placement
			pred_min_index = np.argmin(y_pred)
			my_placement_temp.append(y_test[pred_min_index])
			my_placement_pwr.append(test_comb[pred_min_index, 13])
			data_line.append(y_test[pred_min_index])
			data_line.append(tmp_min_pwr)
			data_line.append(tmp_max_pwr)
			data_line.append(test_comb[pred_min_index, 13])
			data.append(data_line)

		my_ob_diff = np.array(ob_placement_temp) - np.array(my_placement_temp)
		max_diff, max_diff_indices = locate_max(my_ob_diff)
		#print('number of max is ', len(max_diff_indices))
		#print('max_diff is', max_diff, 'combination is ', test_sets[max_diff_indices[0]])
		# best
		best_temp.append(np.mean(best_placement_temp))
		best_pwr.append(np.mean(best_placement_pwr))

		# oblivious
		ob_temp.append(np.mean(ob_placement_temp))
		ob_pwr.append(np.mean(ob_placement_pwr))

		# mine 
		my_temp.append(np.mean(my_placement_temp))
		my_pwr.append(np.mean(my_placement_pwr))
	my_df = pd.DataFrame(data)
	my_df.to_csv('best_worst_RF_placement.csv', index=False, header=False)
	print(np.mean(best_temp), np.mean(best_pwr))
	print(np.mean(ob_temp), np.mean(ob_pwr))
	print(np.mean(my_temp), np.mean(my_pwr))
	err = np.array(err)
	tmp = err.flatten()
	print(np.sqrt(np.mean(tmp**2)))

	print('finish')

def normDataset(data):
	X = np.reshape(data[:,0:12],(np.size(data[:,0]), 12))
	scaler = StandardScaler()
	scaler.fit(X)
	X = scaler.transform(X)

	y = np.reshape(data[:,12:13],(np.size(data[:,0]),1))
	y = np.ravel(y)
	return X, y

def fitLR(X, y):
	lr = LR()
	lr.fit(X, y)
	return lr

def fitMLP(X, y):
	mlp = MLPRegressor(hidden_layer_sizes=(3,), learning_rate_init=0.1, random_state=0, batch_size=100) #, learning_rate='adaptive')
	mlp.fit(X, y)
	return mlp

def fitNN(X, y):
	nn = KNeighborsRegressor(n_neighbors=1, p=2)
	#y = np.ravel(y)
	y_pred = cross_val_predict(nn, X, y, cv=10)
	mse = np.mean((y_pred-y)**2)
	abse = np.mean(np.abs(y_pred-y))
	print("NN: ", mse)
	print("NN: ", abse)
	print("\n")

def fitRF(X, y):
	rf = RandomForestRegressor(n_estimators=120, max_features='auto', max_depth=6, random_state=1)
	rf.fit(X, y)
	return rf

if __name__ == '__main__':
	appU = 'ResourceUsage.csv'
	p1 = "p1.csv"
	p2 = "p2.csv"
	p3 = "p3.csv"
	p4 = "p4.csv"
	usage, p = parseData(appU, p1, p2, p3, p4)
	placement(usage, p)

