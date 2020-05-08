from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer

class Contrast(Layer):
    def call(self, inputs):
        t1 = 0.33 / 2
        t2 = 0.33 + t1
        t3 = 0.66 + t1

        q2 = K.cast((inputs > t1) & (inputs < t2), dtype='float32') * 0.33
        q3 = K.cast((inputs > t2) & (inputs < t3), dtype='float32') * 0.66
        q4 = K.cast((inputs > t3), dtype='float32')

        return q2 + q3 + q4