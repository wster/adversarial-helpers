import tensorflow as tf
from foolbox import TensorFlowModel, accuracy, samples
import foolbox.attacks as fa
import numpy as np


# Performs a PGD attack on a Tensorflow model and returns successful attacks.
def pgd_attack(model, nb_adv, x_test, y_test):
    # Preprocess test data to feed to Foolbox
    x_test, y_test = tf.convert_to_tensor(x_test[:nb_adv], dtype_hint=tf.float32), tf.convert_to_tensor(y_test[:nb_adv], dtype_hint=tf.int64)
    x_test = tf.reshape(x_test, shape=(x_test.shape[0],28,28,1))
    fmodel = TensorFlowModel(model, bounds=(0,1))

    print("Clean accuracy:", accuracy(fmodel, x_test, y_test))
    print("")

    print("Performing PGD attack...")
    attack = fa.LinfPGD()
    epsilons = [0.03, 0.1, 0.3]

    # Report robustness accuracy
    _, _, success = attack(fmodel, x_test, y_test, epsilons=epsilons)
    success_ = success.numpy()

    nb_attacks = len(x_test)
    for i in range(len(epsilons)):
        num_successes = np.count_nonzero(success_[i]==1)
        print("For epsilon = {}, there were {}/{} successful attacks (robustness = {})".format(epsilons[i], num_successes, nb_attacks, round(1.0 - num_successes / nb_attacks, 2)))
