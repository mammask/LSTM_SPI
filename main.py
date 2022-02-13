# script name: main.py
#     purpose: To fit an unbiased LSTM model and forecast the standardized precipitation index
# publication: https://www.mdpi.com/2073-4441/13/18/2531
#      author: Kostas Mammas <mammas_k@live.com>

# Load Python libraries
import pandas as pd
import datetime as dt
import numpy as np
from math import sqrt
import configparser
from keras.optimizers import Adam

# Custom functions
from src.model import LSTM_Modeller

# Import Configuration File
config = configparser.ConfigParser()
config.read('config.ini')

# Initiate model class
modelObject = LSTM_Modeller(config)

# Create iteration path
modelObject.CreateIterationPath()

# Load data
modelObject.DataLoad()

# Compute drought
modelObject.ComputeDrought()

# Reformat data for modelling - Here we can change the look back period --> self.look_back
modelObject.PreprocessData(config)

# Fit LSTM Model
modelObject.FitLSTM(ncells = 2, densnodes = 10,\
    noptimizer = Adam, nepochs = 100, nbatchsize = 2, nlr = 0.001)

# Evaluate LSTM on Test data
overall = modelObject.EvaluateLSTM()

# Generate Reporting
modelObject.MeasurePerformance()
modelObject.performance_test[['Actual', 'Predicted']].plot()