from keras.models import Model
from keras.layers import Input, Conv1D, Activation, Dropout, Lambda, Multiply, Add
from keras.optimizers import Adam

from common.config import *
from multiStepModels.Seq2SeqBase import Seq2SeqBase

class Seq2SeqConvFull(Seq2SeqBase):
    def __init__(self, df):
        Seq2SeqBase.__init__(self, df)
        
        print('Seq2SeqConvFull')
 
    '''
    Building the Model - Training Architecture
    '''
    # extract the last HORIZON time steps as the training target
    def slice(self, x, seq_length):
        return x[:,-seq_length:,:]
    
    def build_training_model(self):
        
        # convolutional operation parameters
        n_filters = S2S_CONVFULL_N_FILTERS # 32 
        filter_width = S2S_CONVFULL_FILTER_WIDTH
        dilation_rates = [2**i for i in range(S2S_CONVFULL_N_DILATIONS)] * 2 
        n_dilation_layers = len(dilation_rates)
        n_dilation_nodes = 2**(S2S_CONVFULL_N_DILATIONS-1)

        # define an input history series and pass it through a stack of dilated causal convolution blocks. 
        history_seq = Input(shape=(None, 1))
        x = history_seq

        skips = []
        for dilation_rate in dilation_rates:        
            # preprocessing - equivalent to time-distributed dense
            x = Conv1D(n_dilation_layers, 1, padding='same', activation='relu')(x) 
            
            # filter convolution
            x_f = Conv1D(filters=n_filters,
                        kernel_size=filter_width, 
                        padding='causal',
                        dilation_rate=dilation_rate)(x)
            
            # gating convolution
            x_g = Conv1D(filters=n_filters,
                        kernel_size=filter_width, 
                        padding='causal',
                        dilation_rate=dilation_rate)(x)
            
            # multiply filter and gating branches
            z = Multiply()([Activation('tanh')(x_f),
                            Activation('sigmoid')(x_g)])
            
            # postprocessing - equivalent to time-distributed dense
            z = Conv1D(n_dilation_layers, 1, padding='same', activation='relu')(z)
            
            # residual connection
            x = Add()([x, z])    
            
            # collect skip connections
            skips.append(z)

        # add all skip connection outputs 
        out = Activation('relu')(Add()(skips))

        # final time-distributed dense layers 
        out = Conv1D(n_dilation_nodes, 1, padding='same')(out)
        out = Activation('relu')(out)
        out = Dropout(.2)(out)
        out = Conv1D(1, 1, padding='same')(out)

        pred_seq_train = Lambda(self.slice, arguments={'seq_length':HORIZON})(out)

        model = Model(history_seq, pred_seq_train)
        model.compile(Adam(), loss='mean_absolute_error')

        print(model.summary())

        return model

    def build_and_train(self):
        self.model = self.build_training_model()
        self.train(
            model=self.model, 
            first_n_samples=120000, 
            batch_size=2**11, 
            epochs=S2S_EPOCHS,
             name='train_history_seq2seq_conv_full')
      