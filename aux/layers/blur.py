import numpy as np

from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer

class Blur(Layer):
    def __init__(self, kernel_size, **kwargs):
        self.kernel_size = (kernel_size, kernel_size)
        self.filters = 1
        super(Blur, self).__init__(**kwargs)

    def build(self, input_shape):
        shape = self.kernel_size + (input_shape[-1], self.filters)
        self.kernel = self.add_weight(shape=shape, initializer=gauss2D)
        super(Blur, self).build(input_shape)

    def call(self, x):
        return K.conv2d(x, self.kernel, padding='same')

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (self.filters,)
    
    def gauss2D(self, shape, dtype=None):
        def nested(shape=(3,3), sigma=0.5):
            m,n = [(ss-1.)/2. for ss in shape]
            y,x = np.ogrid[-m:m+1,-n:n+1]
            h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
            h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
            sumh = h.sum()
            if sumh != 0:
                h /= sumh
            return h
    
        kernel = nested()
        kernel = np.expand_dims(kernel, axis=-1)
        kernel = np.expand_dims(kernel, axis=-1)
        return kernel.astype(dtype)