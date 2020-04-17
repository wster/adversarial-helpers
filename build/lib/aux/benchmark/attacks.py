import tensorflow as tf
from foolbox import TensorFlowModel, accuracy, samples
import foolbox.attacks as fa
import numpy as np


def base(attack, model, images, labels):
    # Preprocess test data to feed to Foolbox
    labels = np.argmax(labels, axis=1) # From categorical to raw labels
    images, labels = tf.convert_to_tensor(images, dtype_hint=tf.float32), tf.convert_to_tensor(labels, dtype_hint=tf.int64)
    fmodel = TensorFlowModel(model, bounds=(-0.3,1.3))

    print("Clean accuracy:", accuracy(fmodel, images, labels))
    print("")

    # Report robustness accuracy
    epsilons = [0.03, 0.1, 0.3]
    _, _, successes = attack(fmodel, images, labels, epsilons=epsilons)
    successes = successes.numpy()

    nb_attacks = len(images)
    for i in range(len(epsilons)):
        num_successes = np.count_nonzero(successes[i]==1)
        print("For epsilon = {}, there were {}/{} successful attacks (robustness = {})".format(epsilons[i], num_successes, nb_attacks, round(1.0 - num_successes / nb_attacks, 2)))


def pgd_attack(model, images, labels):
    """ Evaluates robustness against an L-infinity PGD attack with random restart and 40 steps.
    Args:
        model : Tensorflow model to evaluate.
        images : Clean images that will be turned into adversarial examples
        labels : Labels of the clean images
    """

    print("Performing PGD attack...")
    attack = fa.LinfPGD()
    base(attack, model, images, labels)


def fgsm_attack(model, images, labels):
    """ Evaluates robustness against an L-infinity FGSM attack without random restart.
    Args:
        model : Tensorflow model to evaluate.
        images : Clean images that will be turned into adversarial examples
        labels : Labels of the clean images
    """

    print("Performing FGSM attack...")
    attack = fa.FGSM()
    base(attack, model, images, labels)


def basic_iterative_attack(model, images, labels):
    """ Evaluates robustness against an L-infinity Basic Iterative Attack with 10 steps.
    Args:
        model : Tensorflow model to evaluate.
        images : Clean images that will be turned into adversarial examples
        labels : Labels of the clean images
    """

    print("Performing Basic Iterative Attack...")
    attack = fa.LinfBasicIterativeAttack()
    base(attack, model, images, labels)

"""
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255
x_train = np.expand_dims(x_train, axis=-1)
x_test = x_test.astype('float32') / 255
x_test = np.expand_dims(x_test, axis=-1)

y_train_cat, y_test_cat = to_categorical(y_train), to_categorical(y_test)
inputs = Input(shape=(28,28,1))
flatten = Flatten()(inputs)
outputs = Dense(10, activation='softmax')(flatten)
model = Model(inputs, outputs)
model.compile(optimizer='adam', loss=categorical_crossentropy, metrics=['accuracy'])
model.fit(x_train, y_train_cat)

pgd_attack(model, x_test, y_test_cat)
model.evaluate(x_test, y_test_cat)
"""