"""
Rolling cross-validation for timeseries.
"""
from TRMF_Benchmark.Evaluate import quantile_loss
import numpy as np


def get_slice(data, T_train, T_test, T_start, normalize=True):
    N = len(data)
    # split on train and test
    train = data[:, T_start:T_start+T_train].copy()
    test = data[:, T_start+T_train:T_start+T_train+T_test].copy()
    train=np.array(train, dtype=np.float64)
    test=np.array(test, dtype=np.float64)
    # normalize data
    if normalize:
        mean_train = np.array([])
        std_train = np.array([])
        for i in range(len(train)):
            if (~np.isnan(train[i])).sum() == 0:
                mean_train = np.append(mean_train, 0)
                std_train = np.append(std_train, 0)
            else:
                mean_train = np.append(mean_train, train[i][~np.isnan(train[i])].mean())
                std_train = np.append(std_train, train[i][~np.isnan(train[i])].std())

        std_train[std_train == 0] = 1.

        train -= mean_train.repeat(T_train).reshape(N, T_train)
        train /= std_train.repeat(T_train).reshape(N, T_train)
        test -= mean_train.repeat(T_test).reshape(N, T_test)
        test /= std_train.repeat(T_test).reshape(N, T_test)

    return train, test,mean_train,std_train


def RollingCV(model, data, T_train, T_test, T_step, metric='ND', normalize=True):
    scores = np.array([])
    # for T_start in range(0, data.shape[1]-T_train-T_test+1, T_step): #big data means for scores
    for T_start in range(0, 7*24, 24): #big data means for scores
        train, test,mean,sd = get_slice(data, T_train, T_test, T_start, normalize=normalize) #big train means more time to fit
        model.fit(train)
        test_preds = model.predict(T_test)
        if metric == 'rho':
            scores = np.append(scores, quantile_loss(test_preds, test,mean,sd))

    return scores

