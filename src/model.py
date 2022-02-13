import shutil
import os
import numpy as np
import pandas as pd
import datetime as dt

import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
import rpy2.robjects.packages as rpackages
import rpy2
from rpy2.robjects import r, pandas2ri
pandas2ri.activate()
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.metrics import mean_squared_error, r2_score

# Load R package
droughtr = importr('droughtR')

# Load custom functions
from src.compute_index import compute_drought
from src.manipulate import create_dataset

class LSTM_Modeller:

    def __init__(self, config):

        self.train_size = eval(config['parameters']['train_size'])
        self.valid_size = eval(config['parameters']['valid_size'])
        self.test_size  = eval(config['parameters']['test_size'])
        self.statindex  = eval(config['parameters']['stationarity'])
        self.scale      = config['parameters']['spi_scale']
        self.modelapproach = eval(config['approach']['model'])
        self.city_id = eval(config['location']['city'])
        self.input_data = eval(config['parameters']['input_data'])

        self.train_index = None
        self.valid_index = None
        self.test_index  = None

        self.lool_back = eval(config['model']['look_back'])
        self.time_steps = eval(config['model']['time_steps'])
        self.performance_test = None

        # Define target variable
        if self.statindex is True:
            self.target = 'SPI'
        else:
            self.target = 'NSPI'

    def CreateIterationPath(self):

        # Create iteration Path
        self.iteration = self.modelapproach + "_stationary_" + \
            str(self.statindex) + "_city_" + self.city_id + "_scale_" + \
                str(self.scale)
        
        if os.path.exists(self.iteration):
            shutil.rmtree(self.iteration)
        os.mkdir(self.iteration)

    def DataLoad(self):

        self.df = pd.read_csv(self.input_data)    
        self.df = self.df[self.df['City'] == self.city_id]
        self.df['Date'] = pd.to_datetime(self.df['Time'])
        self.df = self.df.sort_values(by = 'Date', ascending = True)
        self.df['Year'] = self.df['Date'].dt.strftime('%Y')
        self.df['Month'] = self.df['Date'].dt.strftime('%m')
        self.df['Date'] = self.df['Date'].dt.strftime("%Y-%m")
        self.df = self.df.rename(columns = {"Precipitation":"Rainfall", "MaxDate":"MaxTemp", "MinDate":"MinTemp"})
        self.df = self.df.reset_index().drop("index", axis = 1)

        # Split data into train and test
        nrecords = self.df.shape[0]
        self.train_index = int(np.round(nrecords * self.train_size))
        self.valid_index = int(np.round(nrecords * self.valid_size))
        self.test_index  = self.df[self.train_index+self.valid_index:].shape[0]

    def ComputeDrought(self):

        # Obtain training and validation records
        self.train_valid_records = self.df[0:(self.train_index+self.valid_index)].copy()
        # Compute (N)SPI
        self.train_valid_records_spi = compute_drought(self.train_valid_records, self.statindex, self.scale)

        # To add a plot here

        return self.train_valid_records_spi


    def PreprocessData(self, config):

        tseries = self.train_valid_records_spi[['Date', self.target]].copy()
        tseries = tseries.dropna()

        # Obtain the training and validation sets
        train_records = tseries[0:self.train_index].copy()
        valid_records = tseries[self.train_index:(self.train_index+self.valid_index)].copy()

        # Manipulate the univariate ts and obtain the look back period
        self.train_x, self.train_y = create_dataset(train_records[[self.target]].values, look_back=self.lool_back)
        self.valid_x, self.valid_y = create_dataset(valid_records[[self.target]].values, look_back=self.lool_back)

        # Reformat series and define look ahead time step
        self.train_x = np.reshape(self.train_x, (self.train_x.shape[0], self.time_steps, self.train_x.shape[1]))
        self.valid_x = np.reshape(self.valid_x, (self.valid_x.shape[0], self.time_steps, self.valid_x.shape[1]))

    
    def FitLSTM(self, ncells, densnodes, noptimizer, nepochs, nbatchsize, nlr):

        # Fit the LSTM Model
        self.model = Sequential()
        self.model.add(LSTM(ncells, input_shape=(1, self.lool_back), activation= 'relu'))
        self.model.add(Dense(densnodes, activation= 'tanh'))
        self.model.add(Dense(1, activation= 'linear'))
        self.model.compile(loss='mean_squared_error', optimizer=noptimizer(lr=nlr))
        self.model.fit(self.train_x, self.train_y, epochs=nepochs, batch_size=nbatchsize, verbose=2, validation_data=(self.valid_x, self.valid_y))

    def EvaluateLSTM(self):

        preds = []
        actual = []

        # Evaluate the model performance on the test set
        start_id = self.train_index + self.valid_index
        end_id = self.train_index + self.valid_index + self.test_index
        
        for id in range(start_id, end_id):

            # Calculate predicted SPI
            test_records = self.df[0:id].copy()
            test_records['Date'] = pd.to_datetime(test_records['Time'])
            test_records = test_records.sort_values(by = 'Date', ascending = True)
            test_records['Year'] = test_records['Date'].dt.strftime('%Y')
            test_records['Month'] = test_records['Date'].dt.strftime('%m')
            test_records['Date'] = test_records['Date'].dt.strftime("%Y-%m")

            test_records = compute_drought(test_records, self.statindex, self.scale)
            test_records = test_records.dropna()

            test_records_x = test_records.tail(self.lool_back)[self.target].values
            test_records_x = np.reshape(test_records_x, (1,self.time_steps,self.lool_back))
            preds.append(self.model.predict(test_records_x)[0][0])

            # Calculate actual SPI
            actual_records = self.df[0:(id+1)].copy()
            actual_records['Date'] = pd.to_datetime(actual_records['Time'])
            actual_records = actual_records.sort_values(by = 'Date', ascending = True)
            actual_records['Year'] = actual_records['Date'].dt.strftime('%Y')
            actual_records['Month'] = actual_records['Date'].dt.strftime('%m')
            actual_records['Date'] = actual_records['Date'].dt.strftime("%Y-%m")
            actual_records = compute_drought(actual_records, self.statindex, self.scale)
            actual.append(actual_records[self.target].tail(1).values[0])

        mDate = self.df[(self.train_index+self.valid_index):]['Date'].values
        self.performance_test = pd.DataFrame({'Date':mDate,"Actual":actual, "Predicted": preds})

    def MeasurePerformance(self):

        r2 = r2_score(y_true = self.performance_test.Actual,
                      y_pred = self.performance_test.Predicted)

        mse = mean_squared_error(y_true = self.performance_test.Actual,
                                 y_pred = self.performance_test.Predicted
                                 )

        return r2, mse