from common.config import *
from common.TfUtils import load_model
from common.DataStore import DataStore
from common.PlotUtils import plot_forecast, plot_prediction2

import tensorflow as tf
import numpy as np
import pandas as pd
from collections import OrderedDict 
from datetime import datetime

# create an instance of the DataStore
dataStore = DataStore(read_params=True)

# Read data and normalize
print('Read Data')
df = dataStore.read_data()
df_notnormed = df.copy(deep=True)
df = dataStore.transform_df(df=df)

# make a dict with all models to run the predictions
predict_results = OrderedDict()
predict_results[MS_MODEL_NAME_REPEAT_BASELINE] = None
predict_results[MS_MODEL_NAME_MULTISTEP_BASELINE] = None
if (MS_MODEL_LINEAR == True):
    predict_results[MS_MODEL_NAME_LINEAR] = None
if (MS_MODEL_DENSE == True):
    predict_results[MS_MODEL_NAME_DENSE] = None
if (MS_MODEL_CNN == True):
    predict_results[MS_MODEL_NAME_CNN] = None
if (MS_MODEL_RNN == True):
    predict_results[MS_MODEL_NAME_RNN] = None
if (MS_MODEL_AUTOREGRESSIVE_RNN == True):
    predict_results[MS_MODEL_NAME_AUTOREGRESSIVE_RNN] = None

# prepare the dataset
print('Prepare data')
test_df = df[-MS_IN_STEPS:]
test_np = np.array(test_df, dtype=np.float32)
ds = tf.keras.preprocessing.timeseries_dataset_from_array(
    data=test_np,
    targets=None,
    sequence_length=MS_IN_STEPS,
    sequence_stride=1,
    shuffle=True,
    batch_size=1)

# predict
for model_name in predict_results:
    print(f'Predict {model_name}')
    model = load_model(model_name=model_name)
    if model is not None:
        if (model_name == MS_MODEL_NAME_AUTOREGRESSIVE_RNN):
            predict_results[model_name] = model.predict(ds)
        else:
            predict_results[model_name] = model.predict(ds)


# inverse transform
print('Inverse Transform')
#inputs = dataStore.inverse_transform(df=test_df)
for model_name, predict in predict_results.items():
    if (predict is not None):
        predict_results[model_name] = dataStore.inverse_transform(df=predict)

# prepare the prediction we will submit to kaggle for Kaggle
if (PREDICTION_MODEL == 'all'):
    # compute average of all predictions
    predict = np.array([])
    n = 0
    for _, _predict in predict_results.items():
        if (_predict is not None):
            if (predict.size == 0):
                predict = np.zeros(_predict.shape)
            predict = np.add(predict, _predict)
            n += 1
    predict = predict/n
else:
    predict = predict_results[PREDICTION_MODEL]

# transform np array to dataframe
df_predict = pd.DataFrame(
        data=predict[0, :, :],   
        index=pd.date_range("2017-07-31", periods=21),   
        columns=df.columns)
plot_prediction2(df_notnormed, df_predict, colname='series-15')
plot_prediction2(df_notnormed, df_predict, colname='series-19')
plot_prediction2(df_notnormed, df_predict, colname='series-21')
plot_prediction2(df_notnormed, df_predict, colname='series-46')
plot_prediction2(df_notnormed, df_predict, colname='series-69')
kdf = pd.DataFrame(
        data=predict[0, :, :],   
        index=pd.date_range("2017-08-21", periods=21),   
        columns=df.columns)

print('Dump for Kaggle')
now = datetime.now().strftime("%d%m%Y-%H%M%S")
df_forecasts_formatted = dataStore.kaggle_keyvalue(kdf)
df_forecasts_formatted.to_csv(os.path.join(KAGGLE_PATH, 'predict_' + now), index=False)


plot_forecast(df=df_notnormed, df_forecast=kdf, colname='series-15')
plot_forecast(df=df_notnormed, df_forecast=kdf, colname='series-19')
plot_forecast(df=df_notnormed, df_forecast=kdf, colname='series-21')
plot_forecast(df=df_notnormed, df_forecast=kdf, colname='series-46')
plot_forecast(df=df_notnormed, df_forecast=kdf, colname='series-69')
print("DONE!")