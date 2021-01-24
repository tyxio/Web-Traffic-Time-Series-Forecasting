import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from hampel import hampel
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from math import sqrt
from datetime import datetime

from .config import *

def prepare_data(data, remove_outliners=False, mode='validation'):

    # subsetting train and test data
    if (mode == 'validation'):
        # take 20 days for testing
        df_train = data.loc['2015-07-01':'2017-07-30'].copy(deep=True)
        df_test = data.loc['2017-07-31':'2017-08-20'].copy(deep=True)
        df_predictions = pd.DataFrame(index=df_test.index)
    elif (mode == 'kaggle'):
        # use full series to model and predict
        df_train = data.loc['2015-07-01':'2017-08-20'].copy(deep=True)
        # overlap the forecast region
        df_test = pd.DataFrame(index=pd.date_range("2017-08-21", periods=21))
        # construct a new dataframe for the forecasts; create an index
        df_predictions = pd.DataFrame(index=pd.date_range("2017-08-21", periods=21))
    else:
        return

    for (colname, _) in df_train.iteritems():
        if (mode == 'kaggle'):
            df_test[colname] = np.nan
        df_predictions[colname] = np.nan
    
    if (remove_outliners == True):
        for (colname, coldata) in df_train.iteritems():
            df_train[colname] = hampel(df_train[colname], window_size=8, n=3)
    

    return (df_train, df_test, df_predictions)


def test_stationarity(df, label=''):
    is_stationary = 'True'
    dftest = adfuller(df.values)
    if (dftest[1] > 0.05):
        is_stationary = 'False'
    print(label, 'p-value:', dftest[1], 'is stationary:', is_stationary)

# calculate errors


def get_bad_good_predictions(smapes, max_deviation=2):
    mean = np.mean(smapes)
    standard_deviation = np.std(smapes)
    bad_predictions = np.where(
        smapes > mean + max_deviation * standard_deviation)
    good_predictions = np.where(
        smapes < mean - max_deviation * standard_deviation)

    return bad_predictions, good_predictions


def calculate_smape(expected, prediction):
    smape = 100/len(expected) * np.sum(2 * np.abs(prediction -
                                                  expected) / (np.abs(expected) + np.abs(prediction)))
    return smape


def calculate_errors(expected, prediction):
    # Residual Forecast Error
    forecast_errors = [expected[i]-prediction[i] for i in range(len(expected))]
    #print('Forecast Errors: %s' % forecast_errors)
    # Mean Forecast Error (or Forecast Bias)
    bias = sum(forecast_errors) * 1.0/len(expected)
    print('Bias: %f' % bias)
    # Mean Absolute Error
    mae = mean_absolute_error(expected, prediction)
    print('MAE: %f' % mae)
    # Mean Squared Error
    mse = mean_squared_error(expected, prediction)
    print('MSE: %f' % mse)
    # Root Mean Squared Error
    mse = mean_squared_error(expected, prediction)
    rmse = sqrt(mse)
    print('RMSE: %f' % rmse)
    # symmetric mean absolute percentage error
    calculate_smape(expected, prediction)


def plot_estimate2(df_train, df_test, df_forecast, colname, title='Forecasts', config=''):
    plt.figure(figsize=(8, 3))
    plt.plot(df_train.loc['2017-06-01':'2017-08-20'].index,
             df_train.loc['2017-06-01':'2017-08-20', colname], label='Train')
    if (len(df_test) > 0):
        plt.plot(df_test.index, df_test[colname], label='Test')
    plt.plot(df_forecast.index, df_forecast[colname], label=colname)
    plt.legend(loc='best')
    plt.title(title)
    plt.savefig(os.path.join(os.path.join(FIGURES_FOLDER, config), colname))
    plt.close('all')

# dump to Kaggle competition format


def keyvalue(df):
    df["horizon"] = range(1, df.shape[0]+1)
    res = pd.melt(df, id_vars=["horizon"])
    res = res.rename(columns={"variable": "series"})
    res["Id"] = res.apply(
        lambda row: "s" + str(row["series"].split("-")[1]) + "h" + str(row["horizon"]), axis=1)
    res = res.drop(['series', 'horizon'], axis=1)
    res = res[["Id", "value"]]
    res = res.rename(columns={"value": "forecasts"})
    return res

def kaggle_format_and_dump(df, csvfile):
    df_forecasts_formatted = keyvalue(df)
    df_forecasts_formatted.to_csv(csvfile, index=False)
# reports


def report_results(train, test, predictions, smapes, 
        plot_smapes=False, plot_good_predictions=False, plot_bad_predictions=True, config=''):
    now = datetime.now().strftime("%d%m%Y-%H%M%S")
    
    print('Average SMAPE: ', np.mean(smapes))

    bad_predictions, good_predictions = get_bad_good_predictions(smapes)
    print('Bad validations:', bad_predictions)
    print('Good validations:', good_predictions)

    if (plot_bad_predictions == True):
        for col in bad_predictions[0]:
            colname = 'series-' + str(col + 1)
            plot_estimate2(train, test, predictions,
                           colname, title='SMAPE: ' + str(smapes[col]), config=config)
    if (plot_good_predictions == True):
        for col in good_predictions[0]:
            colname = 'series-' + str(col + 1)
            plot_estimate2(train, test, predictions,
                           colname, title='SMAPE: ' + str(smapes[col]), config=config)

    report_smapes(smapes, config)

def report_smapes(smapes, config):
    plt.figure(figsize=(8, 3))
    plt.plot(smapes)
    plt.savefig(os.path.join(os.path.join(FIGURES_FOLDER, config), 'smapes'))
    plt.close('all')

    np.savetxt(os.path.join(os.path.join(SMAPES_FOLDER, config), 'smapes.csv'), smapes, delimiter=",")
