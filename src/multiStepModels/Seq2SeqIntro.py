from keras.models import Model
from keras.layers import Input, LSTM, Dense
from keras.optimizers import Adam
from datetime import timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

from common.config import *
from common.TfUtils import kaggle_keyvalue
from common.PlotUtils import plot_fit_history, plot_forecast
from multiStepModels.Seq2SeqBase import Seq2SeqBase

class Seq2SeqIntro(Seq2SeqBase):
    def __init__(self, df):
        Seq2SeqBase.__init__(self, df)
        
        print('Seq2SeqConvIntro')

    '''
    Building the Model - Training Architecture
    '''

    def build_training_model(self):

        self.latent_dim = S2S_LTSM_HIDDEN_UNITS  # LSTM hidden units
        dropout = .20

        # Define an input series and encode it with an LSTM.
        self.encoder_inputs = Input(shape=(None, 1))
        encoder = LSTM(self.latent_dim, dropout=dropout, return_state=True)
        encoder_outputs, state_h, state_c = encoder(self.encoder_inputs)

        # We discard `encoder_outputs` and only keep the final states. These represent the "context"
        # vector that we use as the basis for decoding.
        self.encoder_states = [state_h, state_c]

        # Set up the decoder, using `encoder_states` as initial state.
        # This is where teacher forcing inputs are fed in.
        self.decoder_inputs = Input(shape=(None, 1))

        # We set up our decoder using `encoder_states` as initial state.
        # We return full output sequences and return internal states as well.
        # We don't use the return states in the training model, but we will use them in inference.
        self.decoder_lstm = LSTM(self.latent_dim, dropout=dropout,
                            return_sequences=True, return_state=True)
        decoder_outputs, _, _ = self.decoder_lstm(self.decoder_inputs,
                                             initial_state=self.encoder_states)

        self.decoder_dense = Dense(1)  # 1 continuous output at each timestep
        decoder_outputs = self.decoder_dense(decoder_outputs)

        # Define the model that will turn
        # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
        model = Model([self.encoder_inputs, self.decoder_inputs], decoder_outputs)
        print(model.summary)

        return model

    def train(self, model):

        first_n_samples = 20000
        batch_size = 2**11
        epochs = 100

        # sample of series from train_enc_start to train_enc_end
        encoder_input_data = self.get_time_block_series(self.series_array, self.date_to_index,
                                                        self.train_enc_start, self.train_enc_end)[:first_n_samples]
        encoder_input_data, encode_series_mean = self.transform_series_encode(
            encoder_input_data)

        # sample of series from train_pred_start to train_pred_end
        decoder_target_data = self.get_time_block_series(self.series_array, self.date_to_index,
                                                         self.train_pred_start, self.train_pred_end)[:first_n_samples]
        decoder_target_data = self.transform_series_decode(
            decoder_target_data, encode_series_mean)

        # lagged target series for teacher forcing
        decoder_input_data = np.zeros(decoder_target_data.shape)
        decoder_input_data[:, 1:, 0] = decoder_target_data[:, :-1, 0]
        decoder_input_data[:, 0, 0] = encoder_input_data[:, -1, 0]

        model.compile(Adam(), loss='mean_absolute_error')
        history = model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
                            batch_size=batch_size,
                            epochs=epochs,
                            validation_split=0.2)

        plot_fit_history(history=history, name='train_history_seq2seq_intro')

    def build_and_train(self):
        model = self.build_training_model()
        self.train(model=model)

    '''
    Building the Model - Inference Architecture
    '''
    def decode_sequence(self, input_seq):
        # Encode the input as state vectors.
        states_value = self.encoder_model.predict(input_seq)

        # Generate empty target sequence of length 1.
        target_seq = np.zeros((1, 1, 1))

        # Populate the first target sequence with end of encoding series pageviews
        target_seq[0, 0, 0] = input_seq[0, -1, 0]

        # Sampling loop for a batch of sequences - we will fill decoded_seq with predictions
        # (to simplify, here we assume a batch of size 1).

        decoded_seq = np.zeros((1, self.pred_steps, 1))

        for i in range(self.pred_steps):

            output, h, c = self.decoder_model.predict([target_seq] + states_value)

            decoded_seq[0, i, 0] = output[0, 0, 0]

            # Update the target sequence (of length 1).
            target_seq = np.zeros((1, 1, 1))
            target_seq[0, 0, 0] = output[0, 0, 0]

            # Update states
            states_value = [h, c]

        return decoded_seq

    def predict_and_plot(self,
             encoder_input_data, decoder_target_data, sample_ind, 
             enc_tail_len=50):
        print(f'Predict {sample_ind}')

        encode_series = encoder_input_data[sample_ind:sample_ind+1, :, :]
        pred_series = self.decode_sequence(encode_series)

        encode_series = encode_series.reshape(-1, 1)
        pred_series = pred_series.reshape(-1, 1)
        target_series = decoder_target_data[sample_ind, :, :1].reshape(-1, 1)

        encode_series_tail = np.concatenate(
            [encode_series[-enc_tail_len:], target_series[:1]])
        x_encode = encode_series_tail.shape[0]
    
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, x_encode+1), encode_series_tail)
        plt.plot(range(x_encode, x_encode+self.pred_steps),
                target_series, color='orange')
        plt.plot(range(x_encode, x_encode+self.pred_steps),
                pred_series, color='teal', linestyle='--')           

        plt.title(
            'Encoder Series Tail of Length %d, Target Series, and Predictions' % enc_tail_len)
        plt.legend(['Encoding Series', 'Target Series', 'Predictions'])

        plt.savefig(os.path.join(FIGURES_PATH, "predict-" + str(sample_ind)))

        return pred_series

    def build_inference_model(self):
        # from our previous model - mapping encoder sequence to state vectors
        self.encoder_model = Model(self.encoder_inputs, self.encoder_states)

        # A modified version of the decoding stage that takes in predicted target inputs
        # and encoded state vectors, returning predicted target outputs and decoder state vectors.
        # We need to hang onto these state vectors to run the next step of the inference loop.
        decoder_state_input_h = Input(shape=(self.latent_dim,))
        decoder_state_input_c = Input(shape=(self.latent_dim,))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

        decoder_outputs, state_h, state_c = self.decoder_lstm(self.decoder_inputs, initial_state=decoder_states_inputs)
        decoder_states = [state_h, state_c]

        decoder_outputs = self.decoder_dense(decoder_outputs)
        self.decoder_model = Model([self.decoder_inputs] + decoder_states_inputs,
                            [decoder_outputs] + decoder_states)        

    def predict(self):
        encoder_input_data = self.get_time_block_series(
            self.series_array, self.date_to_index, self.val_enc_start, self.val_enc_end)
        encoder_input_data, encode_series_mean = self.transform_series_encode(encoder_input_data)

        decoder_target_data = self.get_time_block_series(
            self.series_array, self.date_to_index, self.val_pred_start, self.val_pred_end)
        decoder_target_data = self.transform_series_decode(decoder_target_data, encode_series_mean)

        self.build_inference_model()

        if (PROJECT == 'train_1'):
            self.predict_and_plot(encoder_input_data, decoder_target_data, 100)
            self.predict_and_plot(encoder_input_data, decoder_target_data, 6007)
            self.predict_and_plot(encoder_input_data, decoder_target_data, 33000)
        elif (PROJECT == 'defi3'):
            df_predictions = pd.DataFrame(index=pd.date_range("2017-08-21", periods=21))
            n = 0
            for index, _ in self.df.iterrows():
                pred_series_norm = self.predict_and_plot(encoder_input_data, decoder_target_data, n)
                df_predictions[index] = np.expm1(pred_series_norm + encode_series_mean[n, 0])
                plot_forecast(df=self.df_ori, df_forecast=df_predictions, colname=index)
                n += 1

            print('Dump for Kaggle')
            now = datetime.now().strftime("%d%m%Y-%H%M%S")
            df_forecasts_formatted = kaggle_keyvalue(df_predictions)
            df_forecasts_formatted.to_csv(os.path.join(KAGGLE_PATH, 'predict_' + now), index=False)

            