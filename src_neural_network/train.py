import matplotlib as mpl

from common.config import *
from common.DataStore import DataStore
from common.DataAnalysis import DataAnalysis
from singleStepModels.SingleStepModels import SingleStepModels 
from multiStepModels.MultiStepModels import MultiStepModels
from multiStepModels.Seq2SeqIntro import Seq2SeqIntro
from multiStepModels.Seq2SeqConvIntro import Seq2SeqConvIntro
from multiStepModels.Seq2SeqConvFull import Seq2SeqConvFull
from naiveModels.Naives import Naives


mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False

params = {}
dataStore = DataStore()

# Read data
print('Read Data')
df = dataStore.read_data()

column_indices = {name: i for i, name in enumerate(df.columns)}
num_features = df.shape[1]

# Prepare data for step models
if (SS_RUN_MODELS == True or MS_RUN_MODELS == True):
    # Preprocess the data
    print('Preprocess the Data')
    df = dataStore.preprocess()

    # Run some data analysis
    if (ANALYSIS_FFT):
        dataAnalysis = DataAnalysis(df)
        dataAnalysis.fft()

    # Split the data. We'll use a (70%, 20%, 10%) split for the training, validation, and test sets.
    dataStore.split()

    # Normalize the data
    dataStore.transform()

elif (NAIVE_RUN_MODELS == True):
    # Preprocess the data
    print('Preprocess the Data')
    df = dataStore.preprocess()

'''
Single step models
'''
if (SS_RUN_MODELS == True):
    single_step_models = SingleStepModels(
        dataStore=dataStore,
        column_indices=column_indices,
        label_columns=SS_SINGLE_OUTPUT_FEATURE)

    # build a performance baseline as a point for comparison with the 
    # later more complicated models
    print('Build baseline')
    single_step_models.model_baseline()

    # The simplest trainable model you can apply to this task is to insert 
    # linear transformation between the input and output. 
    if (SS_MODEL_LINEAR):
        print('Linear Model')
        single_step_models.model_linear()

    # Here's a model similar to the linear model, except it stacks 
    # several a few Dense layers between the input and the output:
    if (SS_MODEL_DENSE):
        print('Dense model')
        single_step_models.model_dense()

    if (SS_MODEL_MULTI_INPUT_STEPS_DENSE):
        print('Multi-Steps Dense model')
        single_step_models.model_multi_input_steps_dense()  

    if (SS_MODEL_CNN):
        print('Convolution Neural Network')
        single_step_models.model_cnn() 

    if (SS_MODEL_RNN):
        print('Recurrent Neural Network')
        single_step_models.model_rnn() 

    single_step_models.model_performance()

'''
Multi Step Models
'''
if (MS_RUN_MODELS == True):
    multi_step_models = MultiStepModels(
        dataStore=dataStore,
        column_indices=column_indices)

    # build performance baselines as a points for comparison with the 
    # later more complicated models
    print('Build baselines')
    multi_step_models.model_last_step_baseline()
    multi_step_models.model_repeat_baseline()

    if (MS_MODEL_LINEAR):
        print('Linear Model')
        multi_step_models.model_multi_linear()

    if (MS_MODEL_LINEAR):
        print('Dense model')
        multi_step_models.model_dense()

    if (MS_MODEL_CNN):
        print('Convolution Neural Network')
        multi_step_models.model_cnn() 

    if (MS_MODEL_RNN):
        print('Recurrent Neural Network')
        multi_step_models.model_rnn() 

    if (MS_MODEL_AUTOREGRESSIVE_RNN):
        print('Autoregressive Recurrent Neural Network')
        multi_step_models.model_autoregressive_rnn() 
    

    multi_step_models.model_performance()

'''
Seq2Seq Models
'''
if (S2S_RUN_MODELS == True):
    if (S2S_MODEL_SEQ2SEQ_1):
        print('A relatively simple implementation of the core seq2seq architecture')
        seq2SeqIntro = Seq2SeqIntro(df=df)
        seq2SeqIntro.build_and_train()
        seq2SeqIntro.predict()

    if (S2S_MODEL_SEQ2SEQ_CONV_1):
        print('A convolutional sequence-to-sequence neural network ')
        seq2SeqConvIntro = Seq2SeqConvIntro(df=df)
        seq2SeqConvIntro.build_and_train()
        seq2SeqConvIntro.predict()

    if (S2S_MODEL_SEQ2SEQ_CONV_2):
        print('A convolutional sequence-to-sequence neural network modeled after WaveNet  ')
        seq2SeqConvFull = Seq2SeqConvFull(df=df)
        seq2SeqConvFull.build_and_train()
        seq2SeqConvFull.predict()

    if (S2S_MODEL_SEQ2SEQ_CONV_3):
        print('A convolutional sequence-to-sequence neural network modeled after WaveNet  ')
        print('Adding Exogenous Features to WaveNet  ')
        seq2SeqConvFull = Seq2SeqConvFull(df=df)
        seq2SeqConvFull.build_and_train()
        seq2SeqConvFull.predict()   

'''
Naive Models
'''
if (NAIVE_RUN_MODELS == True):
    if (NAIVE_LAST_3WEEK or NAIVE_LAST_3WEEK_INVERSE):
        print('Copy last week over 3 weeks')
        naive = Naives(df=df)
        naive.predict_copy_last_week()

print('DONE!')