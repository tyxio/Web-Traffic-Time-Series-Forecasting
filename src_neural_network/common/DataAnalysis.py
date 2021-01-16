from .config import *

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime
import tensorflow as tf

class DataAnalysis():
    def __init__(self, df):
        self.df = df

    def fft(self):
        fft = tf.signal.rfft(self.df[ANALYSIS_FFT_COLUMN])
        f_per_dataset = np.arange(0, len(fft))

        n_samples_h = len(self.df[ANALYSIS_FFT_COLUMN])
        hours_per_year = 24*365.2524
        years_per_dataset = n_samples_h/(hours_per_year)

        f_per_year = f_per_dataset/years_per_dataset
        plt.step(f_per_year, np.abs(fft))
        plt.xscale('log')
        plt.ylim(0, 400000)
        plt.xlim([0.1, max(plt.xlim())])
        plt.xticks([1, 365.2524], labels=['1/Year', '1/day'])
        _ = plt.xlabel('Frequency (log scale)')  
        plt.savefig(os.path.join(FIGURES_PATH, 'data_fft'))