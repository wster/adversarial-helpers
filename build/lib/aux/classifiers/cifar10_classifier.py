from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPool2D, Flatten, BatchNormalization, Activation
from tensorflow.keras.regularizers import l2

class CIFAR10Classifier(Model):
    def __init__(self, **kwargs):
        super(CIFAR10Classifier, self).__init__(**kwargs)

        LAMBDA = 1e-4
        
        self.model = Sequential([
            Input(shape=(32,32,3)),
            Conv2D(32, (3,3), padding='same', activation='relu', kernel_regularizer=l2(LAMBDA)),
            BatchNormalization(),

            Conv2D(32, (3,3), padding='same', kernel_regularizer=l2(LAMBDA)),
            MaxPool2D(pool_size=(2,2)),
            Activation('relu'),
            BatchNormalization(),

            Conv2D(64, (3,3), padding='same', activation='relu', kernel_regularizer=l2(LAMBDA)),
            BatchNormalization(),

            Conv2D(64, (3,3), padding='same', kernel_regularizer=l2(LAMBDA)),
            MaxPool2D(pool_size=(2,2)),
            Activation('relu'),
            BatchNormalization(),

            Conv2D(128, (3,3), padding='same', activation='relu', kernel_regularizer=l2(LAMBDA)),
            BatchNormalization(),
            
            Conv2D(128, (3,3), padding='same', activation='relu', kernel_regularizer=l2(LAMBDA)),
            MaxPool2D(pool_size=(2,2)),
            Activation('relu'),
            BatchNormalization(),
            Flatten(),

            Dense(32, activation='relu'),
            Dense(10)
        ])

    def call(self, inputs):
        return self.model(inputs)