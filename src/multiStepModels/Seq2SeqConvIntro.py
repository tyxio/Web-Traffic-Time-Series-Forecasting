from keras.models import Model
from keras.layers import Input, Conv1D, Dense, Dropout, Lambda, concatenate


from common.config import *
from multiStepModels.Seq2SeqBase import Seq2SeqBase

class Seq2SeqConvIntro(Seq2SeqBase):
    def __init__(self, df):
        Seq2SeqBase.__init__(self, df)
        
        print('Seq2SeqConvIntro')

    '''
    Building the Model - Training Architecture
    '''

    def build_training_model(self):

        # convolutional layer parameters
        n_filters = 32 
        filter_width = 2
        dilation_rates = [2**i for i in range(8)] 

        # define an input history series and pass it through a stack of dilated causal convolutions. 
        history_seq = Input(shape=(None, 1))
        x = history_seq

        for dilation_rate in dilation_rates:
            x = Conv1D(filters=n_filters,
                    kernel_size=filter_width, 
                    padding='causal',
                    dilation_rate=dilation_rate)(x)

        x = Dense(128, activation='relu')(x)
        x = Dropout(.2)(x)
        x = Dense(1)(x)

        # extract the last HORIZON time steps as the training target
        def slice(x, seq_length):
            return x[:,-seq_length:,:]

        pred_seq_train = Lambda(slice, arguments={'seq_length':HORIZON})(x)

        model = Model(history_seq, pred_seq_train)
        print(model.summary)

        return model

    def build_and_train(self):
        self.model = self.build_training_model()
        self.train(
            model=self.model, 
            first_n_samples=40000, 
            batch_size=2**11, 
            epochs=S2S_EPOCHS,
            name='train_history_seq2seq_conv_intro')

 
