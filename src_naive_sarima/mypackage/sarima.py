from .utils import calculate_smape
from statsmodels.tsa.statespace.sarimax import SARIMAX

from .config import *

# diable statsmodels warnings
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning, ValueWarning
warnings.simplefilter('ignore', ConvergenceWarning)
warnings.simplefilter('ignore', ValueWarning)

# create a set of sarima configs to try
def sarima_configs(seasonal=[7], test=False):
    models = list()
    if (test == True):
        #cfg = [(2,0,1), (0,1,2,7), 't']
        cfg = [(0,0,0), (0,0,0,0), 'naive5']
        #cfg = [(0,0,0), (0,0,0,0), 'naive2']
        models.append(cfg)
        return models


    # define config lists
    #models.append([(0,0,0), (0,0,0,0), 'naive1'])
    #models.append([(0,0,0), (0,0,0,0), 'naive2'])
    #models.append([(0,0,0), (0,0,0,0), 'naive3'])
    #models.append([(0,0,0), (0,0,0,0), 'naive4'])
    #models.append([(0,0,0), (0,0,0,0), 'naive5'])
    
    p_params = [0, 1, 2]
    d_params = [0, 1]
    q_params = [0, 1, 2]
    t_params = ['n','c','t','ct']
    P_params = [0, 1, 2]
    D_params = [0, 1]
    Q_params = [0, 1, 2]
    m_params = seasonal
    # create config instances
    for p in p_params:
        for d in d_params:
            for q in q_params:
                for t in t_params:
                    for P in P_params:
                        for D in D_params:
                            for Q in Q_params:
                                for m in m_params:
                                    cfg = [(p,d,q), (P,D,Q,m), t]
                                    models.append(cfg)
                                   
    return models

# create a differenced series
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return diff
 
# invert differenced forecast
def inverse_difference(last_ob, value):
	return value + last_ob


# one-step sarima forecast
def sarima_forecast_one_step(history, config):
    order, sorder, trend = config

    # define model
    try:
        model = SARIMAX(history, 
            order=order,
            seasonal_order=sorder,
            trend=trend,
            enforce_stationarity=False,
            enforce_invertibility=False)
        # fit model 
        model_fit = model.fit(disp=False)
        # make 1 step forward prediction
        yhat = model_fit.predict(len(history), len(history))
    except Exception as e:
        print(e)
        yhat = [0.0]

    # inverse

    return yhat[0]

# walk-forward validation for univariate data
def walk_forward_validation(train, test, predictions, config, mode='validation'):
    # seed history with training dataset
    history = [x for x in train]

    # remove seasonality
    if (TRANSFORM_DATA == True):
        history = difference(history, interval=7)
    
   
    # step over each time step in the test set
    for i in range(test.shape[0]):  
        # fit model and make 1-step forecast
        yhat = sarima_forecast_one_step(history, config)  
        # store forecast in list of predictions
        predictions[i] = yhat  
        # add actual observation to history for next loop
        #history.append(df_test[colname][i])
        # Recursive Multi-step Forecast: the prior time step is used as an input for making a prediction on the following time step
        history.append(yhat)

        if (TRANSFORM_DATA == True):
            predictions = [inverse_difference(history[i], 
                predictions[i]) for i in range(len(predictions))]
    
    # estimate prediction error (if using test data)
    smape = 0
    if (mode == 'validation'):
        smape = calculate_smape(test, predictions) 

    return smape
