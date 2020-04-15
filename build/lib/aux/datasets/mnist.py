from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import numpy as np

def load_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.astype('float32') / 255
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = x_test.astype('float32') / 255
    x_test = np.expand_dims(x_test, axis=-1)

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    return (x_train, y_train), (x_test, y_test)