import tensorflow as tf
from foolbox import TensorFlowModel, accuracy, samples
import foolbox.attacks as fa
import numpy as np


def pgd_attack(model, images, labels):
    """ Evaluates robustness against a PGD attack with random restart and 40 iterations.
    Args:
        model : Tensorflow model to evaluate.
        images : Clean images that will be turned into adversarial examples
        labels : Labels of the clean images
    """


    # Preprocess test data to feed to Foolbox
    images, labels = tf.convert_to_tensor(images, dtype_hint=tf.float32), tf.convert_to_tensor(labels, dtype_hint=tf.int64)
    images = tf.reshape(images, shape=(images.shape[0],28,28,1))
    fmodel = TensorFlowModel(model, bounds=(0,1))

    print("Clean accuracy:", accuracy(fmodel, images, labels))
    print("")

    print("Performing PGD attack...")
    attack = fa.LinfPGD()
    epsilons = [0.03, 0.1, 0.3]

    # Report robustness accuracy
    _, _, success = attack(fmodel, images, labels, epsilons=epsilons)
    success_ = success.numpy()

    nb_attacks = len(images)
    for i in range(len(epsilons)):
        num_successes = np.count_nonzero(success_[i]==1)
        print("For epsilon = {}, there were {}/{} successful attacks (robustness = {})".format(epsilons[i], num_successes, nb_attacks, round(1.0 - num_successes / nb_attacks, 2)))

### JUST FOR TESTING ###
def define_model():
    model = tf.keras.Sequential([
          tf.keras.layers.Flatten(input_shape=(28, 28)),
          tf.keras.layers.Dense(128, activation='relu'),
          tf.keras.layers.Dense(10)
    ])
  
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    
    model.fit(X_train, y_train, epochs=10)
    return model

# Download mnist data and split into train and test sets
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

# Preprocess data
X_train, X_test = X_train/255, X_test/255

pgd_attack(define_model(), x_test, y_test)