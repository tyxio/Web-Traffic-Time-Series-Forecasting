from common.config import *
from common.WindowGenerator import WindowGenerator 
from common.DataStore import DataStore
from common.TfUtils import compile_and_fit
from .Baseline import Baseline 

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import tensorflow as tf
import os

class SingleStepModels():
    def __init__(self, 
                 dataStore:DataStore,
                 column_indices=None,
                 label_columns=None):
        # Store the raw data.
        self.dataStore = dataStore
        self.train_df = dataStore.train_df
        self.val_df = dataStore.val_df
        self.test_df = dataStore.test_df
        self.column_indices = column_indices
        self.label_columns = label_columns
        self.num_features = dataStore.train_df.shape[1]

        # Store the performance of the models
        self.val_performance = {}
        self.performance = {}

        # label_columns == 'xxx' ==> Single-output windows 
        # label_columns == None  ==> Multi-output windows
        #  `WindowGenerator` returns all features as labels if you 
        #  don't set the `label_columns` argument.)
        self.single_step_window = self.generate_single_step_window(width=1,label_columns=label_columns)
        self.single_step_wide_window = self.generate_single_step_window(width=24, label_columns=label_columns)

        if (self.label_columns == None):
            self.nunits = self.num_features
        else:
            self.nunits = len(self.label_columns)

        '''
        Create a WindowGenerator that will produce batches 
        of the 3h of inputs and, 1h of labels. Note that the      
        Window's shift parameter is relative to the end of the two windows
        '''
        self.CONV_WIDTH = 3
        self.conv_window = WindowGenerator(
            dataStore=dataStore,
            input_width=self.CONV_WIDTH,
            label_width=1,
            shift=1,
            label_columns=label_columns)


    '''
    This tutorial trains many models, so package the training 
    procedure into a function:
    '''
    MAX_EPOCHS = 20

    def compile_and_fit(self, model, window, patience=2):
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                            patience=patience,
                                                            mode='min')

        model.compile(loss=tf.losses.MeanSquaredError(),
                        optimizer=tf.optimizers.Adam(),
                        metrics=[tf.metrics.MeanAbsoluteError()])

        history = model.fit(window.train, epochs=self.MAX_EPOCHS,
                            validation_data=window.val,
                            callbacks=[early_stopping])
        return history

    '''
    The simplest model you can build on this sort of data is one that predicts a 
    single feature's value, 1 timestep (1h) in the future based only on the 
    current conditions. So start by building models to predict the T (degC) 
    value 1h into the future.
    '''
    def generate_single_step_window(self, width=1, label_columns=None):
        single_step_window = WindowGenerator(
            self.train_df, self.val_df, self.test_df,
            input_width=width, label_width=width, shift=1,           
            label_columns=label_columns
            )
        print(single_step_window)

        for example_inputs, example_labels in single_step_window.train.take(1):
            print(f'Inputs shape (batch, time, features): {example_inputs.shape}')
            print(f'Labels shape (batch, time, features): {example_labels.shape}')

        return single_step_window
    '''
    Before building a trainable model it would be good to have a performance baseline 
    as a point for comparison with the later more complicated models.
    This first task is to predict temperature 1h in the future given the current value
    of all features. The current values include the current temperature.

    So start with a model that just returns the current temperature as the prediction, 
    predicting "No change". This is a reasonable baseline since temperature changes slowly. 
    Of course, this baseline will work less well if you make a prediction further in the future.
    '''
    def model_baseline(self):
        baseline = None
        if (self.label_columns == None):
            baseline = Baseline()
        else: 
            baseline = Baseline(label_index=self.column_indices[self.label_columns[0]])
        baseline.compile(loss=tf.losses.MeanSquaredError(),
                        metrics=[tf.metrics.MeanAbsoluteError()])


        self.val_performance['Baseline'] = baseline.evaluate(self.single_step_window.val)
        self.performance['Baseline'] = baseline.evaluate(self.single_step_window.test, verbose=0) 

        '''
        Create a wider WindowGenerator that generates windows 24h of 
        consecutive inputs and labels at a time.
        The single_step_wide_window doesn't change the way the model operates. The model 
        still makes predictions 1h into the future based on a single 
        input time step. Here the time axis acts like the batch axis: 
        Each prediction is made independently with no interaction 
        between time steps.
        '''
        print('Input shape:', self.single_step_wide_window.example[0].shape)
        print('Output shape:', baseline(self.single_step_wide_window.example[0]).shape)

        self.single_step_wide_window.plot(baseline, plot_name='baseline')

    def model_linear(self):
        '''
        A layers.Dense with no activation set is a linear model. The layer only 
        transforms the last axis of the data from (batch, time, inputs) to 
        (batch, time, units), it is applied independently to every item across 
        the batch and time axes.
        '''

        linear = tf.keras.Sequential([
            tf.keras.layers.Dense(units=self.nunits)])

        print('Input shape:', self.single_step_window.example[0].shape)
        print('Output shape:', linear(self.single_step_window.example[0]).shape)

        history = compile_and_fit(linear, self.single_step_window)

        self.val_performance['Linear'] = linear.evaluate(self.single_step_window.val)
        self.performance['Linear'] = linear.evaluate(self.single_step_window.test, verbose=0)
       
        '''
        Like the baseline model, the linear model can be called on batches of 
        wide windows. Used this way the model makes a set of independent predictions 
        on consecuitive time steps. The time axis acts like another batch axis. 
        There are no interactions between the predictions at each time step.
        '''
        self.single_step_wide_window.plot(linear)

        '''
        One advantage to linear models is that they're relatively simple to interpret. 
        You can pull out the layer's weights, and see the weight assigned to each input:
        '''
        plt.bar(x = range(len(self.train_df.columns)),
            height=linear.layers[0].kernel[:,0].numpy())
        axis = plt.gca()
        axis.set_xticks(range(len(self.train_df.columns)))
        _ = axis.set_xticklabels(self.train_df.columns, rotation=90)
        plt.savefig(os.path.join(FIGURES_PATH, 'linear'))
    
    def model_dense(self):
        '''
        Here's a model similar to the linear model, except it stacks several 
        a few Dense layers between the input and the output:
        '''
        dense = tf.keras.Sequential([
            tf.keras.layers.Dense(units=64, activation='relu'),
            tf.keras.layers.Dense(units=64, activation='relu'),
            tf.keras.layers.Dense(units=self.nunits)
        ])

        history = compile_and_fit(dense, self.single_step_window)

        self.val_performance['Dense'] = dense.evaluate(self.single_step_window.val)
        self.performance['Dense'] = dense.evaluate(self.single_step_window.test, verbose=0)

        self.single_step_wide_window.plot(dense, plot_name='dense')

    def model_multi_input_steps_dense(self):
        '''
        You could train a dense model on a multiple-input-step window 
        by adding a layers.Flatten as the first layer of the model.
        
        The main down-side of this approach is that the resulting model
        can only be executed on input windows of exactly this shape.
        The convolutional models in the next section fix this problem.
        '''
        multi_step_dense = tf.keras.Sequential([
            # Shape: (time, features) => (time*features)
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(units=32, activation='relu'),
            tf.keras.layers.Dense(units=32, activation='relu'),
            tf.keras.layers.Dense(units=self.nunits),
            # Add back the time dimension.
            # Shape: (outputs) => (1, outputs)
            tf.keras.layers.Reshape([1, -1]),
        ])    

        print('Input shape:', self.conv_window.example[0].shape)
        print('Output shape:', multi_step_dense(self.conv_window.example[0]).shape)

        history = compile_and_fit(multi_step_dense, self.conv_window)

        self.val_performance['Multi step dense'] = multi_step_dense.evaluate(self.conv_window.val)
        self.performance['Multi step dense'] = multi_step_dense.evaluate(self.conv_window.test, verbose=0)

        self.conv_window.plot(multi_step_dense, plot_name='multi_input_steps_dense')

    def model_cnn(self):
        conv_model = tf.keras.Sequential([
            tf.keras.layers.Conv1D(filters=32,
                kernel_size=(self.CONV_WIDTH,),
                activation='relu'),
            tf.keras.layers.Dense(units=32, activation='relu'),
            tf.keras.layers.Dense(units=self.nunits),
            ])

        print("Conv model on `conv_window`")
        print('Input shape:', self.conv_window.example[0].shape)
        print('Output shape:', conv_model(self.conv_window.example[0]).shape)

        history = compile_and_fit(conv_model,  self.conv_window)

        self.val_performance['Conv'] =  conv_model.evaluate( self.conv_window.val)
        self.performance['Conv'] =  conv_model.evaluate( self.conv_window.test, verbose=0)

        LABEL_WIDTH = 24
        INPUT_WIDTH = LABEL_WIDTH + (self.CONV_WIDTH - 1)
        wide_conv_window = WindowGenerator(
            self.train_df, self.val_df, self.test_df,
            input_width=INPUT_WIDTH,
            label_width=LABEL_WIDTH,
            shift=1,
            label_columns=self.label_columns)

        
        print("Wide conv window")
        print('Input shape:', wide_conv_window.example[0].shape)
        print('Labels shape:', wide_conv_window.example[1].shape)
        print('Output shape:', conv_model(wide_conv_window.example[0]).shape)

        self.conv_window.plot(conv_model, plot_name='cnn')

    def model_rnn(self):
        self.lstm_model = tf.keras.models.Sequential([
            # Shape [batch, time, features] => [batch, time, lstm_units]
            tf.keras.layers.LSTM(32, return_sequences=True),
            # Shape => [batch, time, features]
            tf.keras.layers.Dense(units=self.nunits)
        ])

        print('Input shape:', self.single_step_wide_window.example[0].shape)
        print('Output shape:', self.lstm_model(self.single_step_wide_window.example[0]).shape)

        history = compile_and_fit(self.lstm_model, self.single_step_wide_window)

        self.val_performance['LSTM'] = self.lstm_model.evaluate(self.single_step_wide_window.val)
        self.performance['LSTM'] = self.lstm_model.evaluate(self.single_step_wide_window.test, verbose=0)

        self.single_step_wide_window.plot(self.lstm_model, plot_name='rnn')

    def model_performance(self):
        x = np.arange(len(self.performance))
        width = 0.3
        metric_name = 'mean_absolute_error'
        metric_index = self.lstm_model.metrics_names.index('mean_absolute_error')
        val_mae = [v[metric_index] for v in self.val_performance.values()]
        test_mae = [v[metric_index] for v in self.performance.values()]

        plt.ylabel('mean_absolute_error [T (degC), normalized]')
        plt.bar(x - 0.17, val_mae, width, label='Validation')
        plt.bar(x + 0.17, test_mae, width, label='Test')
        plt.xticks(ticks=x, labels=self.performance.keys(),
                rotation=45)
        _ = plt.legend() 

        plt.savefig(os.path.join(FIGURES_PATH, 'single_step_performance'))

        for name, value in self.performance.items():
            print(f'{name:12s}: {value[1]:0.4f}')