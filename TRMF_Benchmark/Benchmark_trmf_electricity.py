import numpy as np
import pandas as pd
from TRMF_Benchmark.trmf_pkg.trmf import trmf
from TRMF_Benchmark.trmf_pkg.RollingCV import RollingCV


# The electricity files were too big to push to git, so we split them up
# Note for the real benchmark, these were run on a remote computer with more cores with all the data
T = 1000 #Number of data points
ele1=pd.read_csv('Data/ele1.csv')
ele2=pd.read_csv('Data/ele2.csv')
data=pd.concat([ele1,ele2],axis=1).values[:, -T:]


N = 370 #number of time series
K = 3 #length of embedding dimension
lags = [1]
L = len(lags)

       # TRMF model
lambda_f = 1.
lambda_x = 1.
lambda_w = 1.
eta = 1.
alpha = 1000.
max_iter = 1000


h=24 #prediction length
model = trmf(lags, K, lambda_f, lambda_x, lambda_w, alpha, eta, max_iter)
scores_rho = RollingCV(model, data, T - 24*7 - h, h, T_step=1, metric='rho')
print('rho: ',round(np.array(scores_rho).mean(), 3))
