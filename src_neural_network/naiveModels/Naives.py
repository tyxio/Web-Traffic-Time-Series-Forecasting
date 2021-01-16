from datetime import datetime, timedelta 
import pandas as pd

from common.config import *
from common.PlotUtils import plot_fit_history, plot_forecast, plot_prediction2
from common.TfUtils import kaggle_keyvalue, calculate_smape

class Naives():
    def __init__(self, df):
        print('Naives')
        self.df = df        
   
    def predict_copy_last_week(self):
        now = datetime.now().strftime("%d%m%Y-%H%M%S")
        smapes = pd.Series()
        avsmape = 0
        if (MAKE_PREDICTIONS == True):
            '''
            Predict and plot training results overs the last N (HORIZON) steps
            '''
            m = 7
            df_predictions = []
            if (NAIVE_LAST_3WEEK == True):
                X = self.df.iloc[:-HORIZON, :].iloc[-m:, :]
                df_predictions = pd.concat([X, X, X]) # HORIZON = 3 x m
            elif (NAIVE_LAST_3WEEK_INVERSE == True):
                X = self.df.iloc[:-HORIZON, :].iloc[-m:, :].iloc[::-1]
                Y = self.df.iloc[:-HORIZON, :].iloc[-2*m:-m, :].iloc[::-1]
                Z = self.df.iloc[:-HORIZON, :].iloc[-3*m:-2*m, :].iloc[::-1]
                df_predictions = pd.concat([X, Y, Z]) # HORIZON = 3 x m

            start_test_dt = self.df.index[-1-HORIZON] + timedelta(days=1)
            end_test_dt = start_test_dt + timedelta(days = HORIZON - 1)
            df_predictions.index = pd.date_range(start_test_dt, end_test_dt)           

            n=0
            tsmape = 0
            for series in self.df.columns:
                # plot original data
                smape = plot_prediction2(self.df, df_predictions, series)
                print(f'SMAPE={smape}')
                smapes.at[n+1] = smape
                tsmape += smape
                n += 1
            avsmape = tsmape/n
            print(f'Average SMAPE:{avsmape}')
            smapes.to_csv(os.path.join(KAGGLE_PATH, 'smape_' + now + '.csv'))
        

        if (MAKE_FORECAST == True):
            '''
            Forecast N (HORIZON) future steps
            '''   
            m = 7
            df_predictions = []
            if (NAIVE_LAST_3WEEK == True):
                X = self.df.iloc[-m:, :]
                df_predictions = pd.concat([X, X, X]) # HORIZON = 3 x m
            elif (NAIVE_LAST_3WEEK_INVERSE == True):
                X = self.df.iloc[-m:, :].iloc[::-1]
                Y = self.df.iloc[-2*m:-m, :].iloc[::-1]
                Z = self.df.iloc[-3*m:-2*m, :].iloc[::-1]
                df_predictions = pd.concat([X, Y, Z]) # HORIZON = 3 x m

            start_test_dt = self.df.index[-1] + timedelta(days=1)
            end_test_dt = start_test_dt + timedelta(days = HORIZON - 1)
            df_predictions.index = pd.date_range(start_test_dt, end_test_dt)

            # plot forecast
            for column in self.df.columns:
                plot_forecast(self.df, df_predictions, colname=column)

            # prepare submission to Kaggle
            print('Dump for Kaggle')
            df_forecasts_formatted = kaggle_keyvalue(df_predictions)
            df_forecasts_formatted.to_csv(os.path.join(KAGGLE_PATH, 'predict_' + now + '.csv'), index=False)
            