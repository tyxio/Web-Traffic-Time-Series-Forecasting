from .utils import calculate_smape
import numpy as np
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt

def use_last_value(train, test, predictions, colname, mode='validation'):

    predictions[colname] = train[colname][len(train) - 1]

    smape = 0
    if (mode == 'validation'):
        smape = calculate_smape(test[colname], predictions[colname]) 

    return smape

def moving_average(train, test, predictions, colname, mode='validation'):

    predictions[colname] = train[colname].tail(20).mean()

    smape = 0
    if (mode == 'validation'):
        smape = calculate_smape(test[colname], predictions[colname]) 

    return smape

def simple_exp_smoothing(train, test, predictions, colname, mode='validation'):

    fit2 = SimpleExpSmoothing(np.asarray(train[colname])).fit(smoothing_level=0.6,optimized=False)
    predictions[colname] = fit2.forecast(len(test[colname]))

    smape = 0
    if (mode == 'validation'):
        smape = calculate_smape(test[colname], predictions[colname]) 

    return smape

def holt_linear_trend(train, test, predictions, colname, mode='validation'):

    fit = Holt(np.asarray(train[colname])).fit(smoothing_level = 0.3,smoothing_slope = 0.1)
    predictions[colname] = fit.forecast(len(test[colname]))

    smape = 0
    if (mode == 'validation'):
        smape = calculate_smape(test[colname], predictions[colname]) 

    return smape

def holt_exp_smoothing(train, test, predictions, colname, mode='validation'):

    fit = ExponentialSmoothing(np.asarray(train[colname]) ,seasonal_periods=7 ,trend='add', seasonal='add',).fit()
    predictions[colname] = fit.forecast(len(test[colname]))

    smape = 0
    if (mode == 'validation'):
        smape = calculate_smape(test[colname], predictions[colname]) 

    return smape