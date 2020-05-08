from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer

class Prediction(Layer):
    def __init__(self, threshold, **kwargs):
        self.t = threshold
        self.adv_one_hot = K.one_hot(10, 11)
        super(Prediction, self).__init__(**kwargs)

    def call(self, inputs):
        preds, errors = inputs
        batch_size = K.shape(preds)[0]
        
        # Vector of size batch_size with 1 in each index corresponding to a clean example, and 0 everywhere else
        keep_preds = K.expand_dims(K.cast(errors < self.t, 'float32'), axis=0)
        keep_preds = K.tile(keep_preds, [10, 1])
        keep_preds = K.transpose(keep_preds)

        adv_column = K.reshape(K.cast(errors >= self.t, 'float32'), (batch_size, 1))

        new_preds = preds * keep_preds
        new_preds = K.concatenate([new_preds, adv_column], axis=1)

        return new_preds
