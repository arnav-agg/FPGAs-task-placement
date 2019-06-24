import itertools
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression as LR
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler

indices = range(8)

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

def genDataset(usage, p):
	#possSets = list(itertools.product(indices, repeat=4))
	possSets = list(itertools.permutations(indices, 4))
	print(len(possSets))
	data = []
	tmp = 1
	for s in possSets:
		entry = []
		peakTemp = 0
		sumPow = 0
		for i in s:
			entry.append(usage[i][0])
			entry.append(usage[i][1])
			entry.append(usage[i][2])
		for m in range(4):
			#get peak temp for app n on machine m
			temp = p[m][s[m]][0]
			if peakTemp < temp:  
				peakTemp = temp
			sumPow = sumPow + p[m][s[m]][1]
		entry.append(peakTemp)
		entry.append(sumPow/4.0)

		data.append(entry)
	my_df = pd.DataFrame(data)
	my_df.to_csv('dataset.csv', index=False, header=False)
	print('finish')

def readDataset():
	data = np.array(np.genfromtxt('dataset.csv', delimiter=','))
	#np.random.shuffle(data)
	X = np.reshape(data[:,0:12],(np.size(data[:,0]), 12))
	y = np.reshape(data[:,12:14],(np.size(data[:,0]),2))
	return X, y

def fitLR(X, y):
	lr = LR()
	y_pred = cross_val_predict(lr, X, y, cv=10)
	mse = np.mean((y_pred-y)**2)
	abse = np.mean(np.abs(y_pred-y))
	print(y_pred.shape)
	print("LR: ", mse)
	print("LR: ", abse)
	print("\n")

def fitMLP(X, y):
	mlp = MLPRegressor(hidden_layer_sizes=(5,), learning_rate_init=0.1, random_state=0, batch_size=100) #, learning_rate='adaptive')
	y = np.ravel(y)
	scaler = StandardScaler()
	scaler.fit(X)
	X = scaler.transform(X)
	y_pred = cross_val_predict(mlp, X, y, cv=10)
	mse = np.mean((y_pred-y)**2)
	abse = np.mean(np.abs(y_pred-y))
	print("MLP: ", mse)
	print("MLP: ", abse)
	print("\n")

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
	rf = RandomForestRegressor(n_estimators=100, max_features='auto', max_depth=6, random_state=0)
	y = np.ravel(y)
	y_pred = cross_val_predict(rf, X, y, cv=10)
	mse = np.mean((y_pred-y)**2)
	abse = np.mean(np.abs(y_pred-y))
	print("RF: ", mse)
	print("RF: ", abse)
	print("\n")

if __name__ == '__main__':
	appU = 'ResourceUsage.csv'
	p1 = "p1.csv"
	p2 = "p2.csv"
	p3 = "p3.csv"
	p4 = "p4.csv"
	usage, p = parseData(appU, p1, p2, p3, p4)
	genDataset(usage, p)
	X, y = readDataset()
	'''access temperature data'''
	temp = np.reshape(y[:,0:1],(np.size(y[:,0]),1))
	print(temp)
	scaler = StandardScaler()
	scaler.fit(X)
	X = scaler.transform(X)
	fitLR(X, temp)
	#fitMLP(X, temp)
	#fitNN(X, temp)
	#fitRF(X, temp)
