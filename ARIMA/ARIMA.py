import pandas as pd
from pathlib import Path
from pandas import datetime
from matplotlib import pyplot
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
import warnings
import numpy as np
import pickle
data_loc = Path("../Data/solar.csv")

#Import solar data
solar = pd.read_csv(data_loc, sep=",", index_col=0, parse_dates=True, decimal='.')

training_window = 7 #days
days_predicted = 7 #days
training_steps = 24*training_window
prediction_steps = 24*days_predicted
total_window = training_steps + prediction_steps
rs = 0
station = solar['Power(MW)'][rs:rs+total_window]
#autocorrelation_plot(station0)
training_set = station[0:training_steps]
test_set = station[training_steps:]

def evaluate_arima_model(X, arima_order, train_split):
    train_size = int(len(X)*train_split)
    train, test = X[0:train_size], X[train_size:]
    history = [x for x in train]
    predictions = list()
    for t in range(len(test)):
        model = ARIMA(history, order=arima_order)
        model_fit = model.fit(disp=0)
        yhat = model_fit.forecast()[0]
        predictions.append(yhat)
        history.append(test[t])
    # calculate out of sample error
    error = mean_squared_error(test, predictions)
    return error

def evaluate_models(dataset, p_values, d_values, q_values, train_split):
    dataset = dataset.astype('float32')
    best_score, best_cfg = float("inf"), None
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p,d,q)
                try:
                    mse = evaluate_arima_model(dataset, order, train_split)
                    if mse < best_score:
                        best_score, best_cfg = mse, order
                        print('ARIMA%s MSE=%.3f' % (order,mse))
                except:
                    continue
                print('Best ARIMA%s MSE=%.3f' % (best_cfg, best_score)) 
    return best_score, best_cfg
    
def indicator(predictions, label):
    """Indicator function is a function that returns 1 if label is less than the prediction ."""
    I=np.less(label, predictions)
    return I.astype(int)
def quantile_loss( y_p, y,q=0.5):
    D = (q-indicator(y_p, y))*(y-y_p)
    num = np.sum(D, axis=1)
    den = np.sum(np.abs(y), axis=0)
    Loss = 2*num / den
    return np.average(Loss)

p_vals = range(5,10)
d_vals = range(0,3)
q_vals = range(0,3)
warnings.filterwarnings("ignore")

best_score, best_order = evaluate_models(station, p_vals, d_vals, q_vals, 0.5)


history = [x for x in training_set]
predictions = list()
for t in range(len(test_set)):
 	model = ARIMA(history, order=best_order)
 	model_fit = model.fit(disp=0)
 	output = model_fit.forecast()
 	yhat = output[0]
 	predictions.append(yhat)
 	obs = test_set[t]
 	history.append(obs)
 	print('predicted=%f, expected=%f' % (yhat, obs))
pred_np = np.array(predictions)
test_set_np = np.array(test_set)
error = mean_squared_error(test_set, predictions, squared = False)
rho = quantile_loss(pred_np,test_set_np, q=0.5)

print('Test MSE: %.3f' % error)
print('rho = 0.5 quantile loss %.3f' % rho)
#%% 


# plot
pyplot.plot(test_set_np)
pyplot.plot(pred_np, color='red')
pyplot.show()

with open('predictions.pkl', 'wb') as f:
    pickle.dump(predictions, f)
    



