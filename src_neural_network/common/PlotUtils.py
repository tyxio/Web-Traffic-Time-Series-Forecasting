import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

from .config import *
from .TfUtils import calculate_smape


def plot_forecast(df, df_forecast, colname, title='Forecasts'):
    plt.figure(figsize=(8, 2))
    plt.plot(df.loc['2017-06-01':'2017-08-20'].index,
             df.loc['2017-06-01':'2017-08-20', colname], label='Train')
    
    plt.plot(df_forecast.index, df_forecast[colname], label=colname)
    plt.legend(loc='best')
    plt.title(title + ' ' + colname)
    plt.savefig(os.path.join(FIGURES_PATH, "forecast-" + colname), dpi=70)
    plt.close('all')

def plot_prediction(df, pred_series_norm, series_mean, colname):
    df_predictions = pd.DataFrame(index=pd.date_range(end = "2017-08-20", periods=HORIZON))
    df_predictions[colname] = np.expm1(pred_series_norm + series_mean)
    
    plt.figure(figsize=(8, 2))
    plt.plot(df.loc['2017-06-01':'2017-08-20'].index,
             df.loc['2017-06-01':'2017-08-20', colname], label='Train')
    
    plt.plot(df_predictions.index, df_predictions[colname], label=colname)
    plt.legend(loc='best')
    plt.title('Prediction ' + colname)
    plt.savefig(os.path.join(FIGURES_PATH, "prediction-" + colname), dpi=70)
    plt.close('all')
    
    smape = calculate_smape(df.loc['2017-07-31':'2017-08-20', colname], df_predictions[colname])
    return smape

def plot_prediction2(df, df_predictions, colname):

    plt.figure(figsize=(8, 2))
    plt.plot(df.loc['2017-06-01':'2017-08-20'].index,
             df.loc['2017-06-01':'2017-08-20', colname], label='Train')
    
    plt.plot(df_predictions.index, df_predictions[colname], label=colname)
    plt.legend(loc='best')
    plt.title('Prediction ' + colname)
    plt.savefig(os.path.join(FIGURES_PATH, "prediction-" + colname), dpi=70)
    plt.close('all')

    smape = calculate_smape(df.loc['2017-07-31':'2017-08-20', colname], df_predictions[colname])
    return smape

def plot_random_series(df, n_series):    
    data_start_date = df.columns[1]
    data_end_date = df.columns[-1]
    print('Data ranges from %s to %s' % (data_start_date, data_end_date))

    sample = df.sample(n_series, random_state=8)
    page_labels = sample.index.values.tolist()
    series_samples = sample.loc[:,data_start_date:data_end_date]
    
    plt.figure(figsize=(8,2))
    
    for i in range(series_samples.shape[0]):
        np.log1p(pd.Series(series_samples.iloc[i]).astype(np.float64)).plot(linewidth=1.5)
    
    plt.title('Randomly Selected Wikipedia Page Daily Views Over Time (Log(views) + 1)')
    legend = plt.legend(page_labels, loc='lower center', ncol=3, edgecolor="black")
    legend.get_frame().set_alpha(None)
    legend.get_frame().set_facecolor((0, 0, 1, 0.1))

    plt.savefig(os.path.join(FIGURES_PATH, "input_series"))
    plt.close('all')

def plot_fit_history(history, name):
    plt.figure(figsize=(6,4))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])

    plt.xlabel('Epoch')
    plt.ylabel('Mean Absolute Error Loss')
    plt.title('Loss Over Time')
    plt.legend(['Train','Valid'])
    
    plt.savefig(os.path.join(FIGURES_PATH, name), dpi=70)
    plt.close('all')