import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_forecast(df, df_forecast, colname, title='Forecasts'):
    print(df.head())
    print(df.tail())
    print(df_forecast.tail())
    plt.figure(figsize=(8, 3))
    plt.plot(df.loc['2017-06-01':'2017-08-20'].index,
             df.loc['2017-06-01':'2017-08-20', colname], label='Train')
    
    plt.plot(df_forecast.index, df_forecast[colname], label=colname)
    plt.legend(loc='best')
    plt.title(title)
    plt.show()