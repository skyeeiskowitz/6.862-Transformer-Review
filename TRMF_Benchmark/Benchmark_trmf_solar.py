import numpy as np
import pandas as pd
from TRMF_Benchmark.trmf_pkg.trmf import trmf
from TRMF_Benchmark.trmf_pkg.RollingCV import RollingCV
import random

T = 750 #Number of data points
data=pd.read_csv('Data/solar.csv').values[:,1:].T

N = 137 #number of time series
lags = [1]
L = len(lags)

lambda_f = 1
lambda_x = 1
lambda_w = 1
eta = 1.

alpha = 1000
max_iter = 1000


h=24 #prediction length
#Grid Search: outputs to the console as well as a text file
for i in range(1000):
    with open('best_params_history.txt', 'a') as f:
        K = int(random.uniform(0, 100))
        lambda_f = random.uniform(0, 500)
        lambda_x = random.uniform(0, 500)
        lambda_w = random.uniform(0, 500)
        eta = random.uniform(0, 100)
        alpha = random.uniform(10, 100)
        model = trmf(lags, K, lambda_f, lambda_x, lambda_w, alpha, eta, max_iter)
        scores_rho = RollingCV(model, data, T - 24*7 - h, h, T_step=24, metric='rho')
        print('rho: ',round(np.array(scores_rho).mean(), 3),file=f)
        print('rho: ',round(np.array(scores_rho).mean(), 3))
        print("f: ",lambda_f ,"\nx :",lambda_x,"\nw :",lambda_w,"\neta :",eta,"\nalpha :",alpha,"\nK :",K,"\n``````````````````")
        print("f: ",lambda_f ,"\nx :",lambda_x,"\nw :",lambda_w,"\neta :",eta,"\nalpha :",alpha,"\nK :",K,"\n``````````````````",file=f)
