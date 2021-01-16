from keras.models import Model
from keras.layers import Input, Conv1D, Activation, Dropout, Lambda, Multiply, Add
from keras.optimizers import Adam

from common.config import *
from multiStepModels.Seq2SeqBase import Seq2SeqBase

class Seq2SeqConvFullExog(Seq2SeqBase):
    def __init__(self, df):
        Seq2SeqBase.__init__(self, df)
        
        print('Seq2SeqConvFullExog')