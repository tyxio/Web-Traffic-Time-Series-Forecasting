from tensorflow.python.training.tracking.base import NoRestoreSaveable
from .config import *
from.PlotUtils import plot_random_series

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import datetime
from hampel import hampel
import tensorflow as tf

# Read data


class DataStore():
    def __init__(self, read_params=False):
        self.df = None
        self.train_df = None
        self.val_df = None
        self.test_df = None
        self.date_time = None

        self.train_mean = None
        self.train_std = None

        if (read_params == True):
            self.train_mean = pd.read_pickle(os.path.join(WORK_DATA_PATH, 'train_mean.dat'))
            self.train_std = pd.read_pickle(os.path.join(WORK_DATA_PATH, 'train_std.dat'))

    def read_data(self):
        if (PROJECT == 'jena'):
            self.df = pd.read_csv(CSV_PATH)
            # This tutorial will just deal with hourly predictions, so start
            # by sub-sampling the data from 10 minute intervals to 1h:
            # slice [start:stop:step], starting from index 5 take every 6th record.
            self.df = self.df[5::6]
            # extract the column 'Date Time'
            self.date_time = pd.to_datetime(self.df.pop(
                'Date Time'), format='%d.%m.%Y %H:%M:%S')
        elif (PROJECT == 'defi3'):
            # in this data file there is 1 point/day
            self.df = pd.read_csv(CSV_PATH, index_col="Day", parse_dates=True)
            self.date_time = self.df.index
        elif (PROJECT == 'train_1'):
            # in this data file there is 1 point/day
            self.df = pd.read_csv(CSV_PATH, nrows=DF_NUMBER_OF_SERIES, skiprows=DF_NUMBER_OF_SKIPPED_SERIES)
            self.date_time = self.df.index
        else:
            self.df = None
            self.date_time = None

        #print(self.df.info())
        print(self.df.head())
        print(self.df.shape)

        # Next look at the statistics of the dataset:
        print(self.df.describe().transpose())

        # Let's take a glance at the data
        if (PLOT_INPUT_DATA == True):
            if (PROJECT == 'jena'):
                self.view_data(self.df, self.date_time)
            elif (PROJECT == 'train_1'):
                plot_random_series(self.df, 6)

        return self.df

    def preprocess(self):
        if (PROJECT == 'jena'):
            self.preprocess_jena()
        elif (PROJECT == 'defi3'):
            self.preprocess_defi3()
        return self.df
        
    def preprocess_jena(self):
        '''
        Remove erroneous values for the wind velocity
        '''
        wv = self.df['wv (m/s)']
        bad_wv = wv == -9999.0
        wv[bad_wv] = 0.0

        max_wv = self.df['max. wv (m/s)']
        bad_max_wv = max_wv == -9999.0
        max_wv[bad_max_wv] = 0.0

        '''
        Convert the wind direction and velocity columns to a wind vector:
        '''
        wv = self.df.pop('wv (m/s)')
        max_wv = self.df.pop('max. wv (m/s)')

        # Convert to radians.
        wd_rad = self.df.pop('wd (deg)')*np.pi / 180

        # Calculate the wind x and y components.
        self.df['Wx'] = wv*np.cos(wd_rad)
        self.df['Wy'] = wv*np.sin(wd_rad)

        # Calculate the max wind x and y components.
        self.df['max Wx'] = max_wv*np.cos(wd_rad)
        self.df['max Wy'] = max_wv*np.sin(wd_rad)

        if (PLOT_WIND_VECTORS == True):
            self.plot_wind_vectors(self.df)

        '''
        Convert the time to clear "Time of day" and "Time of year" signals
        '''
        timestamp_s = self.date_time.map(datetime.datetime.timestamp)
        day = 24*60*60
        year = (365.2425)*day
        self.df['Day sin'] = np.sin(timestamp_s * (2 * np.pi / day))
        self.df['Day cos'] = np.cos(timestamp_s * (2 * np.pi / day))
        self.df['Year sin'] = np.sin(timestamp_s * (2 * np.pi / year))
        self.df['Year cos'] = np.cos(timestamp_s * (2 * np.pi / year))

        if (PLOT_TODY_SIGNALS == True):
            self.plot_time_signals(self.df)

        print(self.df.head())
        print(self.df.shape)

    def preprocess_defi3(self):
        data_filtered = self.df.copy(deep=True)

        if (PREPROCESS_REMOVE_OUTLINERS):
            # remove outliners
            print('remove outliners')
            for (colname, coldata) in data_filtered.iteritems():
                data_filtered[colname] = hampel(
                    data_filtered[colname], window_size=8, n=3)

            for s in SERIES_TO_PLOT:
                fig, ax = plt.subplots(figsize=(10, 3))
                ax.plot(self.df[s], label='Original data', marker='.',
                        linestyle='None', markersize=2, color='red')
                ax.plot(data_filtered[s], label='Hampel filtered',
                        marker='.', linestyle='-', markersize=8)
                ax.set_title(s)
                ax.set_ylabel('Traffic')
                ax.legend()
                plt.savefig(os.path.join(FIGURES_PATH, 'data_outliners'))

        self.df = data_filtered

    def split(self):
        if (PROJECT == 'jena' or PROJECT == 'defi3'):
            n = len(self.df)
            if (NO_TEST_DF == True):
                self.train_df = self.df[0:int(n*0.9)]
                self.val_df = self.df[int(n*0.9):]
                self.test_df = self.df[int(n*0.9):]
            else:
                self.train_df = self.df[0:int(n*0.7)]
                self.val_df = self.df[int(n*0.7):int(n*0.9)]
                self.test_df = self.df[int(n*0.9):]

    def transform(self):
        if (PROJECT == 'jena' or PROJECT == 'defi3'):
            # it is the case during training
            self.train_mean = self.train_df.mean()
            self.train_std = self.train_df.std()
            # save to file
            self.train_mean.to_pickle(os.path.join(WORK_DATA_PATH, 'train_mean.dat'))
            self.train_std.to_pickle(os.path.join(WORK_DATA_PATH, 'train_std.dat'))

            # The mean and standard deviation should only be computed using the
            # training data so that the models have no access to the values in
            # the validation and test sets.
            self.train_df = (self.train_df - self.train_mean) / self.train_std
            self.val_df = (self.val_df - self.train_mean) / self.train_std
            self.test_df = (self.test_df - self.train_mean) / self.train_std

            if (PLOT_NORMALIZED_FEATURES == True):
                df_std = (self.df - self.train_mean) / self.train_std
                df_std = df_std.melt(var_name='Column', value_name='Normalized')
                plt.figure(figsize=(12, 6))
                ax = sns.violinplot(x='Column', y='Normalized', data=df_std)
                _ = ax.set_xticklabels(self.df.keys(), rotation=90)
                plt.show()

        
    def transform_df(self, df):
        df = (df - self.train_mean) / self.train_std
        return df

    def inverse_transform(self, df):
        if (type(df) == np.ndarray):
            _df = tf.convert_to_tensor(df, dtype=tf.float32)
            _df = (_df * self.train_std) + self.train_mean
            return _df.numpy()
        else:        
            return (df * self.train_std) + self.train_mean

    def view_data(self, df, date_time):
        if (PROJECT == 'jena'):
            # Here is the evolution of a few features over time.
            plot_cols = ['T (degC)', 'p (mbar)', 'rho (g/m**3)']
            plot_features = df[plot_cols]
            plot_features.index = date_time
            _ = plot_features.plot(subplots=True)

            plot_features = df[plot_cols][:480]
            plot_features.index = date_time[:480]
            _ = plot_features.plot(subplots=True)
            plt.savefig(os.path.join(FIGURES_PATH, 'data_input'))

        elif (PROJECT == 'defi3'):
            axes = df[SERIES_TO_PLOT].plot(
                marker='.', alpha=0.5, linestyle='None', figsize=(11, 9), subplots=True)
            for ax in axes:
                ax.set_ylabel('Traffic')
            plt.savefig(os.path.join(FIGURES_PATH, 'data_input'))

    def plot_wind_vectors(self, df):
        plt.hist2d(df['Wx'], df['Wy'], bins=(50, 50), vmax=400)
        plt.colorbar()
        plt.xlabel('Wind X [m/s]')
        plt.ylabel('Wind Y [m/s]')
        ax = plt.gca()
        ax.axis('tight')
        plt.show()

    def plot_time_signals(self, df):
        plt.plot(np.array(df['Day sin'])[:25])
        plt.plot(np.array(df['Day cos'])[:25])
        plt.xlabel('Time [h]')
        plt.title('Time of day signal')
        plt.show()
        plt.plot(np.array(df['Year sin'])[:])
        plt.plot(np.array(df['Year cos'])[:])
        plt.xlabel('Time [h]')
        plt.title('Time of year signal')
        plt.show()

    def kaggle_keyvalue(self, df):
        df["horizon"] = range(1, df.shape[0]+1)
        res = pd.melt(df, id_vars = ["horizon"])
        res = res.rename(columns={"variable": "series"})
        res["Id"] = res.apply(lambda row: "s" + str(row["series"].split("-")[1]) + "h"+ str(row["horizon"]), axis=1)
        res = res.drop(['series', 'horizon'], axis=1)
        res = res[["Id", "value"]]
        res = res.rename(columns={"value": "forecasts"})
        return res