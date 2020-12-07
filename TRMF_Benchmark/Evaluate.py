import numpy as np

def indicator(predictions, label):
    """Indicator function is a function that returns 1 if label is less than the prediction ."""
    I=np.less(label, predictions)
    return I.astype(int)

def quantile_loss( y_p, y,mean,sd,q=0.5):
    N = y.shape[0]
    h = y.shape[1]
    y_p *= sd.repeat(h).reshape(N, h)
    y_p += mean.repeat(h).reshape(N, h)
    y *= sd.repeat(h).reshape(N, h)
    y += mean.repeat(h).reshape(N, h)
    D = (q-indicator(y_p, y))*(y-y_p)
    num = np.sum(D, axis=1)
    den = np.sum(np.abs(y), axis=1)
    den[den == 0] = 1000000000
    Loss = 2*num / den
    return np.average(Loss)

def indicator(predictions, label):
    """Indicator function is a function that returns 1 if label is less than the prediction ."""
    I=np.less(label, predictions)
    return I.astype(int)

def quantile_loss( y_p, y,q=0.5):


    D = (q-indicator(y_p, y))*(y-y_p)
    num = np.sum(D, axis=1)
    den = np.sum(np.abs(y), axis=1)
    den[den == 0] = 1000000000
    Loss = 2*num / den
    return np.average(Loss)