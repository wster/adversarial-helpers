from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten, Dropout

class MNISTClassifier(Model):
    def __init__(self, **kwargs):
        super(MNISTClassifier, self).__init__(**kwargs)
        
        self.model = Sequential([
            Input(shape=(28,28,1)),
            Conv2D(64, (5,5), activation='relu'),
            Conv2D(64, (5,5), strides=(2,2), activation='relu'),
            Flatten(),
            Dropout(0.25),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(10)
        ])


    def call(self, inputs):
        return self.model(inputs)