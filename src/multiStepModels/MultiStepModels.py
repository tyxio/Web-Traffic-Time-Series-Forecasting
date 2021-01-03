from common.config import *
from common.WindowGenerator import WindowGenerator 
from common.DataStore import DataStore
from common.TfUtils import compile_and_fit, save_model
from .MultiStepLastBaseline import MultiStepLastBaseline 
from .RepeatBaseline import RepeatBaseline
from .FeedBack import FeedBack

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import tensorflow as tf
import os

class MultiStepModels():
    def __init__(self, 
                 dataStore:DataStore,
                 column_indices=None):
        
        # Store the raw data.
        self.train_df = dataStore.train_df
        self.val_df = dataStore.val_df
        self.test_df = dataStore.test_df
        self.column_indices = column_indices
        self.num_features = dataStore.train_df.shape[1]

        # Store the performance of the models
        self.multi_val_performance = {}
        self.multi_performance = {}

        # For the multi-step model, the training data again consists of 
        # hourly samples. However, here, the models will learn to predict 
        # 24h of the future, given 24h of the past.
        self.out_steps = MS_OUT_STEPS
        self.multi_window = WindowGenerator(
            dataStore=dataStore,
            label_width=self.out_steps,
            input_width=MS_IN_STEPS,
            shift=self.out_steps)
        print(self.multi_window)
        #self.multi_window.plot()


    def model_last_step_baseline(self):
        # A simple baseline for this task is to repeat the last 
        # input time step for the required number of output timesteps
        last_baseline = MultiStepLastBaseline()
        last_baseline.compile(loss=tf.losses.MeanSquaredError(),
                            metrics=[tf.metrics.MeanAbsoluteError()])

        self.multi_val_performance['Last'] = last_baseline.evaluate(self.multi_window.val)
        self.multi_performance['Last'] = last_baseline.evaluate(self.multi_window.test, verbose=0)
        
        save_model(model=last_baseline, model_name=MS_MODEL_NAME_MULTISTEP_BASELINE)

        self.multi_window.plot(model=last_baseline, plot_name=MS_MODEL_NAME_MULTISTEP_BASELINE)

    def model_repeat_baseline(self):
        # Since this task is to predict 24h given 24h another simple 
        # approach is to repeat the previous day, assuming tomorrow 
        # will be similar
        repeat_baseline = RepeatBaseline()
        repeat_baseline.compile(loss=tf.losses.MeanSquaredError(),
                                metrics=[tf.metrics.MeanAbsoluteError()])

        self.multi_val_performance['Repeat'] = repeat_baseline.evaluate(self.multi_window.val)
        self.multi_performance['Repeat'] = repeat_baseline.evaluate(self.multi_window.test, verbose=0)

        save_model(model=repeat_baseline, model_name=MS_MODEL_NAME_REPEAT_BASELINE)

        self.multi_window.plot(model=repeat_baseline, plot_name=MS_MODEL_NAME_REPEAT_BASELINE)

    def model_multi_linear(self):
        multi_linear_model = tf.keras.Sequential([
            # Take the last time-step.
            # Shape [batch, time, features] => [batch, 1, features]
            tf.keras.layers.Lambda(lambda x: x[:, -1:, :]),
            # Shape => [batch, 1, out_steps*features]
            tf.keras.layers.Dense(MS_OUT_STEPS*self.num_features,
                                kernel_initializer=tf.initializers.zeros()),
            # Shape => [batch, out_steps, features]
            tf.keras.layers.Reshape([MS_OUT_STEPS, self.num_features])
        ])

        history = compile_and_fit(multi_linear_model, self.multi_window)
        
        self.multi_val_performance['Linear'] = multi_linear_model.evaluate(self.multi_window.val)
        self.multi_performance['Linear'] = multi_linear_model.evaluate(self.multi_window.test, verbose=0)

        save_model(model=multi_linear_model, model_name=MS_MODEL_NAME_LINEAR)

        self.multi_window.plot(model=multi_linear_model, plot_name=MS_MODEL_NAME_LINEAR) 

    def model_dense(self):
        multi_dense_model = tf.keras.Sequential([
            # Take the last time step.
            # Shape [batch, time, features] => [batch, 1, features]
            tf.keras.layers.Lambda(lambda x: x[:, -1:, :]),
            # Shape => [batch, 1, dense_units]
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(512, activation='relu'),
            # Shape => [batch, out_steps*features]
            tf.keras.layers.Dense(MS_OUT_STEPS*self.num_features,
                                kernel_initializer=tf.initializers.zeros()),
            # Shape => [batch, out_steps, features]
            tf.keras.layers.Reshape([MS_OUT_STEPS, self.num_features])
        ])

        history = compile_and_fit(multi_dense_model, self.multi_window)

        self.multi_val_performance['Dense'] = multi_dense_model.evaluate(self.multi_window.val)
        self.multi_performance['Dense'] = multi_dense_model.evaluate(self.multi_window.test, verbose=0)
        
        save_model(model=multi_dense_model, model_name=MS_MODEL_NAME_DENSE)
        
        self.multi_window.plot(multi_dense_model, plot_name=MS_MODEL_NAME_DENSE)

    def model_cnn(self):
        CONV_WIDTH = 3
        multi_conv_model = tf.keras.Sequential([
            # Shape [batch, time, features] => [batch, CONV_WIDTH, features]
            tf.keras.layers.Lambda(lambda x: x[:, -CONV_WIDTH:, :]),
            # Shape => [batch, 1, conv_units]
            tf.keras.layers.Conv1D(256, activation='relu', kernel_size=(CONV_WIDTH)),
            # Shape => [batch, 1,  out_steps*features]
            tf.keras.layers.Dense(MS_OUT_STEPS*self.num_features,
                                kernel_initializer=tf.initializers.zeros()),
            # Shape => [batch, out_steps, features]
            tf.keras.layers.Reshape([MS_OUT_STEPS, self.num_features])
        ])

        history = compile_and_fit(multi_conv_model, self.multi_window)

        self.multi_val_performance['Conv'] = multi_conv_model.evaluate(self.multi_window.val)
        self.multi_performance['Conv'] = multi_conv_model.evaluate(self.multi_window.test, verbose=0)
               
        save_model(model=multi_conv_model, model_name=MS_MODEL_NAME_CNN)
   
        self.multi_window.plot(multi_conv_model, plot_name=MS_MODEL_NAME_CNN)

    def model_rnn(self):
        self.multi_lstm_model = tf.keras.Sequential([
            # Shape [batch, time, features] => [batch, lstm_units]
            # Adding more `lstm_units` just overfits more quickly.
            tf.keras.layers.LSTM(32, return_sequences=False),
            # Shape => [batch, out_steps*features]
            tf.keras.layers.Dense(MS_OUT_STEPS*self.num_features,
                                kernel_initializer=tf.initializers.zeros()),
            # Shape => [batch, out_steps, features]
            tf.keras.layers.Reshape([MS_OUT_STEPS, self.num_features])
        ])

        history = compile_and_fit(self.multi_lstm_model, self.multi_window)

        self.multi_val_performance['LSTM'] = self.multi_lstm_model.evaluate(self.multi_window.val)
        self.multi_performance['LSTM'] = self.multi_lstm_model.evaluate(self.multi_window.test, verbose=0)

        save_model(model=self.multi_lstm_model, model_name=MS_MODEL_NAME_RNN)

        self.multi_window.plot(self.multi_lstm_model, plot_name=MS_MODEL_NAME_RNN)

    def model_autoregressive_rnn(self):
        feedback_model = FeedBack(num_features=self.num_features, units=32, out_steps=MS_OUT_STEPS)
        
        prediction, state = feedback_model.warmup(self.multi_window.example[0])
        prediction.shape

        print('Output shape (batch, time, features): ', feedback_model(self.multi_window.example[0]).shape)

        history = compile_and_fit(feedback_model, self.multi_window)

        self.multi_val_performance['AR LSTM'] = feedback_model.evaluate(self.multi_window.val)
        self.multi_performance['AR LSTM'] = feedback_model.evaluate(self.multi_window.test, verbose=0)
        
        save_model(model=feedback_model, model_name=MS_MODEL_NAME_AUTOREGRESSIVE_RNN)

        self.multi_window.plot(feedback_model, plot_name=MS_MODEL_NAME_AUTOREGRESSIVE_RNN)

    def model_performance(self):
        x = np.arange(len(self.multi_performance))
        width = 0.3

        metric_name = 'mean_absolute_error'
        metric_index = self.multi_lstm_model.metrics_names.index('mean_absolute_error')
        val_mae = [v[metric_index] for v in self.multi_val_performance.values()]
        test_mae = [v[metric_index] for v in self.multi_performance.values()]
        
        plt.figure(figsize=(12, 8))
        plt.bar(x - 0.17, val_mae, width, label='Validation')
        plt.bar(x + 0.17, test_mae, width, label='Test')
        plt.xticks(ticks=x, labels=self.multi_performance.keys(),
                rotation=45)
        plt.ylabel(f'MAE (average over all times and outputs)')
        _ = plt.legend()
        
        plt.savefig(os.path.join(FIGURES_PATH, MS_MODEL_PERFORMANCE_NAME))
        
        for name, value in self.multi_performance.items():
            print(f'{name:8s}: {value[1]:0.4f}')

