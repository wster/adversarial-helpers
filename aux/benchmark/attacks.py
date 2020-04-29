import tensorflow as tf
import foolbox.attacks as fa
import numpy as np

from foolbox import TensorFlowModel, accuracy, samples
from tensorflow.keras.utils import to_categorical

def base(attack, model, images, labels, epsilons, bounds):
    # Preprocess test data to feed to Foolbox
    labels = np.argmax(labels, axis=1) # From categorical to raw labels
    images, labels = tf.convert_to_tensor(images, dtype_hint=tf.float32), tf.convert_to_tensor(labels, dtype_hint=tf.int64)
    fmodel = TensorFlowModel(model, bounds=bounds)

    print("Clean accuracy:", accuracy(fmodel, images, labels))
    print("")

    # Report robustness accuracy
    _, imgs, successes = attack(fmodel, images, labels, epsilons=epsilons)
    successes = successes.numpy()
    success_imgs, success_labels = [], []

    nb_attacks = len(images)
    for i in range(len(epsilons)):
        success_idxs = successes[i] == 1

        success_imgs.append(imgs[i][success_idxs])
        categorical_labels = to_categorical(labels.numpy()[success_idxs])
        success_labels.append(categorical_labels)

        num_successes = np.count_nonzero(success_idxs)
        print("For epsilon = {}, there were {}/{} successful attacks (robustness = {})".format(epsilons[i], num_successes, nb_attacks, round(1.0 - num_successes / nb_attacks, 2)))
    
    return success_imgs, success_labels 

def pgd_attack(model, images, labels, epsilons=[0.03, 0.1, 0.3], bounds=(0,1)):
    """ Evaluates robustness against an L-infinity PGD attack with random restart and 40 steps.
    Args:
        model : Tensorflow model to evaluate.
        images : Clean images that will be turned into adversarial examples
        labels : Labels of the clean images
    """

    print("Performing PGD attack...")
    attack = fa.LinfPGD()
    return base(attack, model, images, labels, epsilons, bounds)


def fgsm_attack(model, images, labels, epsilons=[0.03, 0.1, 0.3], bounds=(0,1)):
    """ Evaluates robustness against an L-infinity FGSM attack without random restart.
    Args:
        model : Tensorflow model to evaluate.
        images : Clean images that will be turned into adversarial examples
        labels : Labels of the clean images
    """

    print("Performing FGSM attack...")
    attack = fa.FGSM()
    return base(attack, model, images, labels, epsilons, bounds)


def basic_iterative_attack(model, images, labels, epsilons=[0.03, 0.1, 0.3], bounds=(0,1)):
    """ Evaluates robustness against an L-infinity Basic Iterative Attack with 10 steps.
    Args:
        model : Tensorflow model to evaluate.
        images : Clean images that will be turned into adversarial examples
        labels : Labels of the clean images
    """

    print("Performing Basic Iterative Attack...")
    attack = fa.LinfBasicIterativeAttack()
    return base(attack, model, images, labels, epsilons, bounds)


def cw_attack(model, images, labels, epsilons=[0.03, 0.1, 0.3], bounds=(0,1)):
    """ Evaluates robustness against an L2 CW attack .
    Args:
        model : Tensorflow model to evaluate.
        images : Clean images that will be turned into adversarial examples
        labels : Labels of the clean images
    """

    print("Performing CW Attack...")
    attack = fa.L2CarliniWagnerAttack()
    return base(attack, model, images, labels, epsilons, bounds)



### JUST FOR TESTING ###
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPool2D, Flatten
# Download mnist data and split into train and test sets
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()


# Preprocess data
x_train = x_train.astype('float32') / 255
x_train = np.expand_dims(x_train, axis=-1)
x_test = x_test.astype('float32') / 255
x_test = np.expand_dims(x_test, axis=-1)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

def define_model():
    model = Sequential([
            Input(shape=(28,28,1)),
            Conv2D(32, (5,5), activation='relu', padding='same'),
            MaxPool2D(),
            Conv2D(64, (5,5), activation='relu', padding='same'),
            MaxPool2D(),
            Flatten(),
            Dense(1024, activation='relu'),
            Dense(10, activation='softmax')
    ])
  
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    
    model.fit(x_train, y_train, epochs=10)
    return model

cw_attack(define_model(), x_test, y_test)