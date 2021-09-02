import numpy as np
from sklearn.metrics import mean_squared_error


def rmspe(y_true, y_pred):
    return np.sqrt(np.mean(np.square((y_true - y_pred) / y_true)))


def lgb_rmse(y_true, y_pred):
    return "rmse", mean_squared_error(y_true, y_pred, squared=False), False


def lgb_rmspe(y_true, y_pred):
    return "rmspe", rmspe(y_true, y_pred), False
