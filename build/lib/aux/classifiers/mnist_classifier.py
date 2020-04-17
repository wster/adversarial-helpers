from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten

class MNISTClassifier(Model):
    def __init__(self, inputs, **kwargs):
        super(MNISTClassifier, self).__init__(**kwargs)
        
        self.model = Sequential([
            inputs,
            Conv2D(32, (5,5), activation='relu', padding='same'),
            MaxPool2D(),
            Conv2D(64, (5,5), activation='relu', padding='same'),
            MaxPool2D(),
            Flatten(),
            Dense(1024, activation='relu'),
            Dense(10, activation='softmax')
        ])

    def call(self, inputs):
        return self.model(inputs)