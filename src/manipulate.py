import pandas as pd
import numpy as np

def univariate_data(dataset, start_idx , end_idx , history_size, target_size):
    data = []
    labels = []
    start_idx  = start_idx + history_size
    if end_idx is None:
        end_idx = len(dataset)- target_size
    for i in range(start_idx , end_idx):
        idxs = range(i-history_size , i)
        data.append(np.reshape(dataset[idxs] , (history_size, 1)))
        labels.append(dataset[i+target_size])
    return np.array(data), np.array(labels)

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return np.array(dataX), np.array(dataY)