from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
from random import randint

class Noise(Layer):
    def __init__(self, minval, maxval, shape, **kwargs):
        self.minval = minval
        self.maxval = maxval
        self.shape = shape
        super(Noise, self).__init__(**kwargs)

    def build(self, input_shape):
        shape = list(self.shape)
        shape.insert(0, 60000)
        shape = tuple(shape)

        self.noise = K.random_uniform(shape=shape, minval=self.minval, maxval=self.maxval)
        super(Noise, self).build(input_shape)

    def call(self, x):
        rand_int = randint(0, 60000)
        noise = self.noise[rand_int]
        return x + noise

    def compute_output_shape(self, input_shape):
        return input_shape