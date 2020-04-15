import tensorflow as tf
from foolbox import TensorFlowModel, accuracy, samples
import foolbox.attacks as fa
import numpy as np


def base(attack, model, images, labels):
    # Preprocess test data to feed to Foolbox
    images, labels = tf.convert_to_tensor(images, dtype_hint=tf.float32), tf.convert_to_tensor(labels, dtype_hint=tf.int64)
    images = tf.reshape(images, shape=(images.shape[0],28,28,1))
    fmodel = TensorFlowModel(model, bounds=(0,1))

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
