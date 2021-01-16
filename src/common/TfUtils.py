from .config import *

import os
import shutil
import pandas as pd
import numpy as np
import tensorflow as tf

def compile_and_fit(model, window, patience=5):
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=patience,
        mode='min')

    model.compile(loss=tf.losses.MeanSquaredError(),
                    optimizer=tf.optimizers.Adam(),
                    metrics=[tf.metrics.MeanAbsoluteError()])

    history = model.fit(window.train, epochs=MS_MAX_EPOCHS,
                        validation_data=window.val,
                        callbacks=[early_stopping]
                        )
    return history

def save_model(model:tf.keras.Model, model_name:str):
    path = os.path.join(MODELS_PATH, model_name)
    try:
        if (os.path.exists(path)):
            shutil.rmtree(path)
        model.save(path)
    except OSError as e:
        print("Error: %s : %s" % (path, e.strerror))

def load_model(model_name:str) -> tf.keras.Model:
    path = os.path.join(MODELS_PATH, model_name)
    if (os.path.exists(path) == False):
        print("Error: %s does not exist" % (path))
    
    return tf.keras.models.load_model(os.path.join(MODELS_PATH, model_name))

def kaggle_keyvalue(df):
    df["horizon"] = range(1, df.shape[0]+1)
    res = pd.melt(df, id_vars = ["horizon"])
    res = res.rename(columns={"variable": "series"})
    res["Id"] = res.apply(lambda row: "s" + str(row["series"].split("-")[1]) + "h"+ str(row["horizon"]), axis=1)
    res = res.drop(['series', 'horizon'], axis=1)
    res = res[["Id", "value"]]
    res = res.rename(columns={"value": "forecasts"})
    return res

def calculate_smape(expected, prediction):
    smape = 100/len(expected) * np.sum(2 * np.abs(prediction -
           expected) / (np.abs(expected) + np.abs(prediction)))

    return smape