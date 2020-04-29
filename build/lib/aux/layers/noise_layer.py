from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer

class Noise(Layer):
    def __init__(self, minval, maxval, **kwargs):
        self.minval = minval
        self.maxval = maxval
        super(Noise, self).__init__(**kwargs)

    def call(self, x):
        noise = K.random_uniform(shape=K.shape(x), minval=self.minval, maxval=self.maxval)
        return x + noise

    def compute_output_shape(self, input_shape):
        return input_shape