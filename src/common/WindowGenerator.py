from .config import *
from .DataStore import DataStore

import numpy as np
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import tensorflow as tf


class WindowGenerator():
    def __init__(self, 
            input_width, 
            label_width, 
            shift, 
            dataStore:DataStore,                           
            label_columns=None):
        # Store the raw data
        self.dataStore = dataStore
        self.train_df = dataStore.train_df
        self.val_df = dataStore.val_df
        self.test_df = dataStore.test_df

        # Work out the label column indices.
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in
                                          enumerate(label_columns)}
        self.column_indices = {name: i for i, name in
                               enumerate(self.train_df.columns)}

        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[
            self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[
            self.labels_slice]

    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'])

    def split_window(self, features):
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
            labels = tf.stack(
                [labels[:, :, self.column_indices[name]]
                    for name in self.label_columns],
                axis=-1)

        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels

    def plot(self, model=None, plot_col=PLOT_COLUMN, max_subplots=3, plot_name='figure', normed=False):
        inputs_norm = None
        labels_norm = None
        if (normed == True):
            inputs, labels = self.example
        else:
            inputs_norm, labels_norm = self.example
            inputs = self.dataStore.inverse_transform(df=inputs_norm)
            labels = self.dataStore.inverse_transform(df=labels_norm)

        plt.figure(figsize=(12, 8))
        plot_col_index = self.column_indices[plot_col]
        max_n = min(max_subplots, len(inputs))
        for n in range(max_n):
            plt.subplot(3, 1, n+1)
            plt.ylabel(f'{plot_col} [normed]')
            plt.plot(self.input_indices, inputs[n, :, plot_col_index],
                    label='Inputs', marker='.', zorder=-10)

            if self.label_columns:
                label_col_index = self.label_columns_indices.get(plot_col, None)
            else:
                label_col_index = plot_col_index

            if label_col_index is None:
                continue

            plt.scatter(self.label_indices, labels[n, :, label_col_index],
                        edgecolors='k', label='Labels', c='#2ca02c', s=64)
            if model is not None:
                if (normed == True):
                    predictions = model(inputs)
                else:
                    predictions_norm = model(inputs_norm)
                    predictions = self.dataStore.inverse_transform(df=predictions_norm)
                
                plt.scatter(self.label_indices, predictions[n, :, label_col_index],
                            marker='X', edgecolors='k', label='Predictions',
                            c='#ff7f0e', s=64)

            if n == 0:
                plt.legend()

        plt.xlabel('Time [h]')
        
        plt.savefig(os.path.join(FIGURES_PATH, plot_name))

    def make_dataset(self, data):
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=True,
            batch_size=32)

        ds = ds.map(self.split_window)
        
        return ds

    @property
    def train(self):
        return self.make_dataset(self.train_df)

    @property
    def val(self):
        return self.make_dataset(self.val_df)

    @property
    def test(self):
        return self.make_dataset(self.test_df)

    @property
    def example(self):
        """Get and cache an example batch of `inputs, labels` for plotting."""
        result = getattr(self, '_example', None)
        if result is None:
            # No example batch was found, so get one from the `.train` dataset
            result = next(iter(self.train))
            # And cache it for next time
            self._example = result
        return result


def test_WindowGenerator(train_df, val_df, test_df):
    w1 = WindowGenerator(
        train_df=train_df, val_df=val_df, test_df=test_df,
        input_width=24, label_width=1, shift=24, label_columns=['T (degC)'])
    print(w1)

    w2 = WindowGenerator(
        train_df=train_df, val_df=val_df, test_df=test_df,
        input_width=6, label_width=1, shift=1, label_columns=['T (degC)'])                 
    print(w2)

    '''
    Typically data in TensorFlow is packed into arrays where the outermost index 
    is across examples (the "batch" dimension). The middle indices are the 
    "time" or "space" (width, height) dimension(s). The innermost indices are the features.
    
    The code below takes a batch of 3, 7-timestep windows, with 19 features 
    at each time step. It split them into a batch of 6-timestep, 19 feature inputs, 
    and a 1-timestep 1-feature label. The label only has one feature because 
    the WindowGenerator was initialized with label_columns=['T (degC)']. 
    Initially this tutorial will build models that predict single output labels.
    '''
    # Stack three slices, the length of the total window:
    example_window = tf.stack([np.array(train_df[:w2.total_window_size]),
                            np.array(train_df[100:100+w2.total_window_size]),
                            np.array(train_df[200:200+w2.total_window_size])])

    example_inputs, example_labels = w2.split_window(example_window)

    print('All shapes are: (batch, time, features)')
    print(f'Window shape: {example_window.shape}')
    print(f'Inputs shape: {example_inputs.shape}')
    print(f'labels shape: {example_labels.shape}')

    # The Dataset.element_spec property tells you the structure, dtypes and shapes of the dataset elements.
    w2.train.element_spec

    # Iterating over a Dataset yields concrete batches:
    for example_inputs, example_labels in w2.train.take(1):
        print(f'Inputs shape (batch, time, features): {example_inputs.shape}')
        print(f'Labels shape (batch, time, features): {example_labels.shape}')

    w2.plot()
