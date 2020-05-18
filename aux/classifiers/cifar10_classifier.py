from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPool2D, Flatten, BatchNormalization, Activation, Dropout
from tensorflow.keras.regularizers import l2

class CIFAR10Classifier(Model):
    def __init__(self, **kwargs):
        super(CIFAR10Classifier, self).__init__(**kwargs)

        self.model = Sequential([
            Input(shape=(32,32,3)),
            Conv2D(64, (5,5), padding='same', activation='relu'),
            Conv2D(128, (5,5), padding='same', activation='relu'),
            Conv2D(256, (5,5), padding='same', activation='relu', dilation_rate=2),
            Dropout(0.25),
            Dense(512, activation='relu'),
            Dropout(0.5),
            Dense(256, activation='relu'),
            Dense(10, activation='relu'),
        ])

    def call(self, inputs):
        return self.model(inputs)