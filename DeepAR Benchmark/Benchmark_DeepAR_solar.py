from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from datetime import datetime, timedelta
import pandas as pd
import math
import numpy as np
import random
from tqdm import trange
import argparse
import logging
import DeepAR.utils as utils

import torch
from torch import nn
import torch.optim as optim
from torch.utils.data.sampler import RandomSampler
from tqdm import tqdm

from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile

from math import sqrt
from pandas import read_csv, DataFrame
from scipy import stats

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
#sys.path.append(r'C:\Users\jwigm\Documents\GitHub\6.862-Transformer')
import DeepAR_train 
import DeepAR.Model.net as net
from DeepAR.dataloader import *



logger = logging.getLogger('DeepAR.Train')
parser = argparse.ArgumentParser()


def prep_data(data, covariates, data_start, data_param, train = True):
    
    #Originally used global variables for this:
    window_size, stride_size, num_covariates, total_time, num_series = data_param
    save_name = 'solar'
    save_path = os.path.join('DeepAR/Processed_Data', save_name)
    #print("train: ", train)
    time_len = data.shape[0] #total hours of training data = total hours (5832) - testing hours (48)
    #print("time_len: ", time_len)
    input_size = window_size-stride_size
    windows_per_series = np.full((num_series), (time_len-input_size) // stride_size)
    #print("windows pre: ", windows_per_series.shape)
    if train: windows_per_series -= (data_start+stride_size-1) // stride_size
    #print("data_start: ", data_start.shape)
    #print(data_start)
    #print("windows: ", windows_per_series.shape)
    #print(windows_per_series)
    total_windows = np.sum(windows_per_series)
    x_input = np.zeros((total_windows, window_size, 1 + num_covariates + 1), dtype='float32')
    label = np.zeros((total_windows, window_size), dtype='float32')
    v_input = np.zeros((total_windows, 2), dtype='float32')
    #cov = 3: ground truth + age + day_of_week + hour_of_day + num_series
    #cov = 4: ground truth + age + day_of_week + hour_of_day + month_of_year + num_series
    count = 0
    if not train:
        covariates = covariates[-time_len:]
    for series in trange(num_series):
        cov_age = stats.zscore(np.arange(total_time-data_start[series]))
        if train:
            covariates[data_start[series]:time_len, 0] = cov_age[:time_len-data_start[series]]
        else:
            covariates[:, 0] = cov_age[-time_len:]
        for i in range(windows_per_series[series]):
            if train:
                window_start = stride_size*i+data_start[series]
            else:
                window_start = stride_size*i
            window_end = window_start+window_size
            '''
            print("x: ", x_input[count, 1:, 0].shape)
            print("window start: ", window_start)
            print("window end: ", window_end)
            print("data: ", data.shape)
            print("d: ", data[window_start:window_end-1, series].shape)
            '''
            x_input[count, 1:, 0] = data[window_start:window_end-1, series]
            x_input[count, :, 1:1+num_covariates] = covariates[window_start:window_end, :]
            x_input[count, :, -1] = series
            label[count, :] = data[window_start:window_end, series]
            nonzero_sum = (x_input[count, 1:input_size, 0]!=0).sum()
            if nonzero_sum == 0:
                v_input[count, 0] = 0
            else:
                v_input[count, 0] = np.true_divide(x_input[count, 1:input_size, 0].sum(),nonzero_sum)+1
                x_input[count, :, 0] = x_input[count, :, 0]/v_input[count, 0]
                if train:
                    label[count, :] = label[count, :]/v_input[count, 0]
            count += 1
    prefix = os.path.join(save_path, 'train_' if train else 'test_')
    np.save(prefix+'data_'+save_name, x_input)
    np.save(prefix+'v_'+save_name, v_input)
    np.save(prefix+'label_'+save_name, label)
    print('Solar Data Processed!')

def gen_covariates(times, num_covariates):
    covariates = np.zeros((times.shape[0], num_covariates))
    for i, input_time in enumerate(times):
        covariates[i, 1] = input_time.weekday()
        covariates[i, 2] = input_time.hour
        covariates[i, 3] = input_time.month
    for i in range(1,num_covariates):
        covariates[:,i] = stats.zscore(covariates[:,i])
    return covariates[:, :num_covariates]

def visualize(data, week_start, window_size):
    x = np.arange(window_size)
    f = plt.figure()
    plt.plot(x, data[week_start:week_start+window_size], color='b')
    f.savefig("visual.png")
    plt.close()
    
    
def pre_process_solar():
    #name = 'solar.csv'
    #save_name = 'solar'
    window_size = 192#
    stride_size = 12 #
    num_covariates = 4
    
    train_start = '2006-01-01 00:00:00'
    train_end = '2006-08-24 23:00:00' 
    test_start = '2006-08-17 00:00:00' #need additional day as given info
    test_end = '2006-08-31 23:00:00'
    
    
    data_frame = pd.read_csv('solar.csv',sep=",", index_col=0, parse_dates=True, decimal=',' )
    data_frame.fillna(0, inplace=True)
    covariates = gen_covariates(data_frame[train_start:test_end].index, num_covariates)
    train_data = data_frame[train_start:train_end].values
    test_data = data_frame[test_start:test_end].values
    data_start = (train_data!=0).argmax(axis=0) #find first nonzero value in each time series
    total_time = data_frame.shape[0] #total number of hours in the dataset
    num_series = data_frame.shape[1] #total number of sites i.e. 137
    data_param = [window_size, stride_size, num_covariates, total_time, num_series]
    prep_data(train_data, covariates, data_start, data_param, train = True)
    prep_data(test_data, covariates, data_start, data_param, train=False)
    return data_frame, covariates
    
    
    

if __name__ == '__main__':
    
    #Preprocssing the solar data
    print('Preprocessing solar data:')
    # data, covariates = pre_process_solar()
    sampling = True 
    model_dir = 'DeepAR/Model'
    json_path = 'DeepAR/Model/params.json'
    data_dir = 'DeepAR/Processed_Data/solar'
    dataset = 'solar'
    params = utils.Params(json_path)
    params.relative_metrics = True
    params.sampling =  True
    params.model_dir = model_dir
    params.plot_dir = os.path.join(model_dir, 'figures')
    workers = 0
    # create missing directories
    try:
        os.mkdir(params.plot_dir)
    except FileExistsError:
        pass

    # use GPU if available
    cuda_exist = torch.cuda.is_available()  #boolean whether or not GPU is available
    # Set random seeds for reproducible experiments if necessary
    if cuda_exist:
        params.device = torch.device('cuda')
        # torch.cuda.manual_seed(240)
        logger.info('Using Cuda...')
        model = net.Net(params).cuda()
    else:
        params.device = torch.device('cpu')
        # torch.manual_seed(230)
        logger.info('Not using cuda...')
        model = net.Net(params)

    utils.set_logger(os.path.join(model_dir, 'train.log'))
    logger.info('Loading the datasets...')
    
    train_set = TrainDataset(data_dir, dataset, params.num_class)
    test_set = TestDataset(data_dir, dataset, params.num_class)
    sampler = WeightedSampler(data_dir, dataset) # Use weighted sampler instead of random sampler
    train_loader = DataLoader(train_set, batch_size=params.batch_size, sampler=sampler, num_workers = workers)
    test_loader = DataLoader(test_set, batch_size=params.predict_batch, sampler=RandomSampler(test_set), num_workers = workers)
    logger.info('Loading complete.')

    logger.info(f'Model: \n{str(model)}')
    optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)

    # fetch loss function'
    loss_fn = net.loss_fn

    # Train the model
    logger.info('Starting training for {} epoch(s)'.format(params.num_epochs))
    DeepAR_train.train_and_evaluate(model,
                        train_loader,
                        test_loader,
                        optimizer,
                        loss_fn,
                        params,
                        None)
                        #args.restore_file)

