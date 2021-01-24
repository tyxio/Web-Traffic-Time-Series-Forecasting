from datetime import timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

from keras.optimizers import Adam

from common.config import *
from common.PlotUtils import plot_fit_history, plot_forecast, plot_prediction, plot_random_series
from common.TfUtils import kaggle_keyvalue

class Seq2SeqBase():
    def __init__(self, df):
        print('Seq2SeqConvFull')

        self.df_ori = df.copy(deep=True)
        
        if (PROJECT == 'defi3'):
            # transpose
            self.df = df.transpose()
            print(self.df.head())
            print(self.df.tail())
        else:
            self.df = df

        print(self.df.info())
        print(self.df.head())
        print(self.df.shape)

        plot_random_series(self.df, 6)

        # Train and Validation Series Partioning
        self.pred_steps = HORIZON
        pred_length = timedelta(self.pred_steps)

        self.data_start_date=self.df.columns[1]
        self.data_end_date=self.df.columns[-1]
        first_day = pd.to_datetime(self.data_start_date)
        last_day = pd.to_datetime(self.data_end_date)

        self.val_pred_start = last_day - pred_length + timedelta(1)
        self.val_pred_end = last_day

        self.train_pred_start = self.val_pred_start - pred_length
        self.train_pred_end = self.val_pred_start - timedelta(days=1)

        enc_length = self.train_pred_start - first_day

        self.train_enc_start = first_day
        self.train_enc_end = self.train_enc_start + enc_length - timedelta(1)

        self.val_enc_start = self.train_enc_start + pred_length
        self.val_enc_end = self.val_enc_start + enc_length - timedelta(1)

        print('Train encoding:', self.train_enc_start, '-', self.train_enc_end)
        print('Train prediction:', self.train_pred_start,
              '-', self.train_pred_end, '\n')
        print('Val encoding:', self.val_enc_start, '-', self.val_enc_end)
        print('Val prediction:', self.val_pred_start, '-', self.val_pred_end)

        print('\nEncoding interval:', enc_length.days)
        print('Prediction interval:', pred_length.days)

        self.date_to_index = pd.Series(index=pd.Index([pd.to_datetime(c) for c in self.df.columns[1:]]),
                                  data=[i for i in range(len(self.df.columns[1:]))])

        self.series_array = self.df[self.df.columns[1:]].values

    '''
    Formatting the Data for Modeling
    '''

    def get_time_block_series(self, series_array, date_to_index, start_date, end_date):

        inds = date_to_index[start_date:end_date]
        return series_array[:, inds]

    def transform_series_encode(self, series_array):

        series_array = np.log1p(np.nan_to_num(series_array))  # filling NaN with 0
        series_mean = series_array.mean(axis=1).reshape(-1, 1)
        series_array = series_array - series_mean
        series_array = series_array.reshape(
            (series_array.shape[0], series_array.shape[1], 1))

        return series_array, series_mean

    def transform_series_decode(self, series_array, encode_series_mean):

        series_array = np.log1p(np.nan_to_num(
            series_array))  # filling NaN with 0
        series_array = series_array - encode_series_mean
        series_array = series_array.reshape(
            (series_array.shape[0], series_array.shape[1], 1))

        return series_array

    '''
    Train the model
    '''

    def train(self, model, first_n_samples, batch_size, epochs, name):
        
        # sample of series from train_enc_start to train_enc_end  
        encoder_input_data = self.get_time_block_series(self.series_array, self.date_to_index, 
                                    self.train_enc_start, self.train_enc_end)[:first_n_samples]
        encoder_input_data, encode_series_mean = self.transform_series_encode(encoder_input_data)

        # sample of series from train_pred_start to train_pred_end 
        decoder_target_data = self.get_time_block_series(self.series_array, self.date_to_index, 
                                    self.train_pred_start, self.train_pred_end)[:first_n_samples]
        decoder_target_data = self.transform_series_decode(decoder_target_data, encode_series_mean)

        # we append a lagged history of the target series to the input data, 
        # so that we can train with teacher forcing
        lagged_target_history = decoder_target_data[:,:-1,:1]
        encoder_input_data = np.concatenate([encoder_input_data, lagged_target_history], axis=1)

        model.compile(Adam(), loss='mean_absolute_error')
        history = model.fit(encoder_input_data, decoder_target_data,
                            batch_size=batch_size,
                            epochs=epochs,
                            validation_split=0.2)

        plot_fit_history(history=history, name=name)

    '''
    Building the Model - Inference Architecture
    '''
    
    def predict_sequence(self, input_sequence):

        history_sequence = input_sequence.copy()
        pred_sequence = np.zeros((1,self.pred_steps,1)) # initialize output (pred_steps time steps)  
        
        for i in range(self.pred_steps):
            
            # record next time step prediction (last time step of model output) 
            last_step_pred = self.model.predict(history_sequence)[0,-1,0]
            pred_sequence[0,i,0] = last_step_pred
            
            # add the next time step prediction to the history sequence
            history_sequence = np.concatenate([history_sequence, 
                                            last_step_pred.reshape(-1,1,1)], axis=1)

        return pred_sequence

    '''
    Prediction
    '''

    def predict_series(self, encoder_input_data, sample_ind):
        print(f'Predict {sample_ind}')

        encode_series = encoder_input_data[sample_ind:sample_ind+1,:,:] 
        pred_series = self.predict_sequence(encode_series)
        
        encode_series = encode_series.reshape(-1,1)
        pred_series = pred_series.reshape(-1,1)   
        
        return encode_series, pred_series  
    
    def plot_prediction(self, 
            encode_series, pred_series, target_series, 
            sample_ind, enc_tail_len=50):

        encode_series_tail = np.concatenate([encode_series[-enc_tail_len:],target_series[:1]])
        x_encode = encode_series_tail.shape[0]

        plt.figure(figsize=(10,6))   
        
        plt.plot(range(1,x_encode+1),encode_series_tail)
        plt.plot(range(x_encode,x_encode+self.pred_steps),target_series,color='orange')
        plt.plot(range(x_encode,x_encode+self.pred_steps),pred_series,color='teal',linestyle='--')
        
        plt.title('Encoder Series Tail of Length %d, Target Series, and Predictions' % enc_tail_len)
        plt.legend(['Encoding Series','Target Series','Predictions'])
        plt.savefig(os.path.join(FIGURES_PATH, "predict_norm-" + str(sample_ind)))
        plt.close('all')

    def predict_and_plot(self,
             encoder_input_data, decoder_target_data, sample_ind, 
             enc_tail_len=50):
        encode_series, pred_series = self.predict_series(
                encoder_input_data, sample_ind)

        target_series = decoder_target_data[sample_ind,:,:1].reshape(-1,1) 
        
        self.plot_prediction(encode_series, pred_series, target_series, sample_ind)

        return pred_series

    def predict(self):
        now = datetime.now().strftime("%d%m%Y-%H%M%S")
        smapes = pd.Series()
        avsmape = 0
        if (MAKE_PREDICTIONS == True):
            '''
            Predict and plot training results overs the last N (HORIZON) steps
            '''
            encoder_input_data = self.get_time_block_series(
                self.series_array, self.date_to_index, self.val_enc_start, self.val_enc_end)
            encoder_input_data, encode_series_mean = self.transform_series_encode(encoder_input_data)

            decoder_target_data = self.get_time_block_series(
                self.series_array, self.date_to_index, self.val_pred_start, self.val_pred_end)
            decoder_target_data = self.transform_series_decode(decoder_target_data, encode_series_mean)

            if (PROJECT == 'train_1'):
                self.predict_and_plot(encoder_input_data, decoder_target_data, 10)
                self.predict_and_plot(encoder_input_data, decoder_target_data, 20)
                self.predict_and_plot(encoder_input_data, decoder_target_data, 50)
            
            elif (PROJECT == 'defi3'):
                n=0
                tsmape = 0
                for index, _ in self.df.iterrows():
                    # predict and plot normalized data
                    pred_series_norm = self.predict_and_plot(encoder_input_data, decoder_target_data, n)
                    # plot original data
                    smape = plot_prediction(self.df_ori, pred_series_norm, encode_series_mean[n, 0], index)
                    print(f'{n} SMAPE={smape}')
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
            df_forecasts = pd.DataFrame(index=pd.date_range("2017-08-21", periods=HORIZON))

            encoder_input_data = self.get_time_block_series(
                self.series_array, self.date_to_index, 
                self.val_enc_start + timedelta(days=HORIZON), 
                self.val_enc_end + timedelta(days=HORIZON))
            encoder_input_data, encode_series_mean = self.transform_series_encode(encoder_input_data)
            
            n = 0
            for index, _ in self.df.iterrows():
                # forecast
                print(f'Forecast {index}')
                _, pred_series_norm = self.predict_series(encoder_input_data, n)
                df_forecasts[index] = np.expm1(pred_series_norm + encode_series_mean[n, 0])

                # plot forecast
                plot_forecast(self.df_ori, df_forecasts, colname=index)
                n += 1

            # prepare submission to Kaggle
            print('Dump for Kaggle')            
            df_forecasts_formatted = kaggle_keyvalue(df_forecasts)
            df_forecasts_formatted.to_csv(os.path.join(KAGGLE_PATH, 'predict_' + now + '.csv'), index=False)
            
            # dump variables
            print(f'HORIZON:{HORIZON}')
            print(f'S2S_EPOCHS:{S2S_EPOCHS}')
            print(f'S2S_CONVFULL_N_FILTERS:{S2S_CONVFULL_N_FILTERS}')
            print(f'S2S_CONVFULL_N_DILATIONS:{S2S_CONVFULL_N_DILATIONS}')
            print(f'S2S_CONVFULL_FILTER_WIDTH:{S2S_CONVFULL_FILTER_WIDTH}')
            print(f'Average SMAPE:{avsmape}')