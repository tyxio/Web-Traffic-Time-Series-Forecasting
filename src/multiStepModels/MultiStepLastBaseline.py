from common.config import *

import tensorflow as tf

class MultiStepLastBaseline(tf.keras.Model):
    def call(self, inputs):
        return tf.tile(inputs[:, -1:, :], [1, MS_OUT_STEPS, 1])