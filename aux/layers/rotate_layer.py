from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer

class Rotate(Layer):
    def __init__(self, k, **kwargs):
        super(Rotate, self).__init__(**kwargs)
        self.k = k % 4

    def call(self, x):
        transposed = K.transpose(x)
        permuted = K.permute_dimensions(transposed, (3,1,2,0))
        
        if self.k == 0:
            return x
        elif self.k == 1:
            return K.reverse(permuted, 2)
        elif self.k == 2:
            return K.reverse(K.reverse(x, 1), 2)
        elif self.k == 3:
            return K.reverse(permuted, 1)

    def compute_output_shape(self, input_shape):
        return input_shape