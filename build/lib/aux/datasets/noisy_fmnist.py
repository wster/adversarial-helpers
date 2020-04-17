from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.utils import to_categorical
import numpy as np

def load_data(low, high):
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    x_train = x_train.astype('float32') / 255
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = x_test.astype('float32') / 255
    x_test = np.expand_dims(x_test, axis=-1)

    noise_train = np.random.uniform(low=low, high=high, size=x_train.shape)
    noise_test = np.random.uniform(low=low, high=high, size=x_test.shape)

    x_train = np.clip(x_train + noise_train, 0, 1)
    x_test = np.clip(x_test + noise_test, 0, 1)

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    return (x_train, y_train), (x_test, y_test)