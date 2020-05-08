from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
from tensorflow.keras.losses import mean_squared_error

class BestReconstruction(Layer):
    def __init__(self, puvae, num_classes, **kwargs):
        self.puvae = puvae
        self.num_classes = num_classes
        self.eye = K.eye(num_classes)

        super(BestReconstruction, self).__init__(**kwargs)

    def call(self, inputs):
        batch_size = K.shape(inputs)[0]
        images = K.repeat_elements(inputs, self.num_classes, axis=0)
        labels = K.tile(self.eye, [batch_size, 1])

        _, _, reconstructions = self.puvae([images, labels])
        errors = K.mean(K.sqrt(mean_squared_error(images, reconstructions)), axis=(1,2))
        errors = K.reshape(errors, (batch_size, self.num_classes))

        best_idxs = K.argmin(errors) + K.arange(0, batch_size, dtype='int64') * self.num_classes
        best_reconstructions = K.gather(reconstructions, best_idxs)

        return best_reconstructions, K.min(errors, axis=1)