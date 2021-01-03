import os 

PROJECT='jena' # jena, defi3
INPUT_DATA_PATH = os.path.join('data', 'input')
WORK_DATA_PATH = os.path.join('data', 'work')
REPORTS_PATH = os.path.join('reports', PROJECT)
FIGURES_PATH = os.path.join(REPORTS_PATH, 'figures')
MODELS_PATH = os.path.join('models', PROJECT)
KAGGLE_PATH =  os.path.join(REPORTS_PATH, 'kaggle')

if (PROJECT == 'jena'):
    CSV_PATH = os.path.join(INPUT_DATA_PATH, 'jena_climate_2009_2016.csv')
    ANALYSIS_FFT_COLUMN='T (degC)'
    PLOT_COLUMN='T (degC)'
    PLOT_WIND_VECTORS=False
    PLOT_TODY_SIGNALS=False
elif (PROJECT == 'defi3'):
    CSV_PATH = os.path.join(INPUT_DATA_PATH, 'defi3.csv')
    PREPROCESS_REMOVE_OUTLINERS=True
    SERIES_TO_PLOT = ["series-10", "series-20", "series-30", "series-47"]  
    ANALYSIS_FFT_COLUMN='series-10'
    PLOT_COLUMN='series-10'
    

PLOT_INPUT_DATA=False
PLOT_NORMALIZED_FEATURES=False
ANALYSIS_FFT=False
TEST_WINDOW_GENERATOR=False

# ===========================================
# Single Step Models
SS_RUN_MODELS=False  # run single step models? If False, all SS_ configurations below are ignored

SS_SINGLE_OUTPUT_FEATURE=None # None=all features, else ['T (degC)']

SS_MODEL_LINEAR=True
SS_MODEL_DENSE=True
SS_MODEL_MULTI_INPUT_STEPS_DENSE=True
SS_MODEL_CNN=True
SS_MODEL_RNN=True

# ===========================================
# Multi Step Models
MS_RUN_MODELS=True  # run multi step models? If False, all SS_ configurations below are ignored

if (PROJECT == 'jena'):
    MS_IN_STEPS = 24
    MS_OUT_STEPS = 24
    MS_MAX_EPOCHS = 20
elif (PROJECT == 'defi3'):
    MS_IN_STEPS = 21
    MS_OUT_STEPS = 21
    MS_MAX_EPOCHS = 100

# True to run model, else False
MS_MODEL_LINEAR=True
MS_MODEL_DENSE=True
MS_MODEL_CNN=True
MS_MODEL_RNN=True
MS_MODEL_AUTOREGRESSIVE_RNN=True

MS_MODEL_NAME_REPEAT_BASELINE='multi_baseline_repeat'
MS_MODEL_NAME_MULTISTEP_BASELINE='multi_baseline_last_step'
MS_MODEL_NAME_LINEAR='multi_linear'
MS_MODEL_NAME_DENSE='multi_dense'
MS_MODEL_NAME_CNN='multi_cnn'
MS_MODEL_NAME_RNN='multi_lstm'
MS_MODEL_NAME_AUTOREGRESSIVE_RNN='multi_feedback_lstm'
MS_MODEL_PERFORMANCE_NAME='multi_step_performance'

PREDICTION_MODEL=MS_MODEL_NAME_DENSE