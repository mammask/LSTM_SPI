# Load python libraries
import pandas as pd
import numpy as np
import datetime as dt
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
import rpy2.robjects.packages as rpackages
import rpy2
from rpy2.robjects import r, pandas2ri
pandas2ri.activate()
from rpy2.robjects.conversion import localconverter

# Load R libraries
droughtr = importr('droughtR')

def compute_drought(df, stationary, spiscale):
    
    # Compute the standardized precipitation index in the training set
    df = droughtr.computenspi(monthlyRainfall = df, stationary = stationary, spiScale = float(spiscale))
    
    # Revert back to a pandas dataframe format manually
    dfP = pd.DataFrame()
    dfcolnames = df._get_colnames()
    for i in range(1, len(dfcolnames)+1):
        getvalues = []
        for j in df.rx2(i):
            getvalues.append(j)
        if dfcolnames[i-1] == 'Trend' or dfcolnames[i-1] == 'ecdfm':
            dfP[dfcolnames[i-1]] = np.where(getvalues == "NA_integer_", None, getvalues)
        else:
            dfP[dfcolnames[i-1]] = getvalues


    # Recover date
    dfP['Date'] = dfP.apply(lambda x: str(x['Year']) + '-' + str(x['Month'] + '-01'), axis = 1)
    dfP['Date'] = pd.to_datetime(dfP['Date'])
    dfP['Date'] = dfP['Date'].dt.strftime('%Y-%m')
    
    return dfP