from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPool2D, Flatten

class MNISTClassifier(Model):
    def __init__(self, **kwargs):
        super(MNISTClassifier, self).__init__(**kwargs)
        
        self.model = Sequential([
            Input(shape=(28,28,1)),
            Conv2D(32, (5,5), activation='relu', padding='same'),
            MaxPool2D(),
            Conv2D(64, (5,5), activation='relu', padding='same'),
            MaxPool2D(),
            Flatten(),
            Dense(1024, activation='relu'),
            Dense(10)
        ])

    def call(self, inputs):
        return self.model(inputs)