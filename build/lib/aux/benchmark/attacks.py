import tensorflow as tf
from foolbox import TensorFlowModel, accuracy, samples
import foolbox.attacks as fa
import numpy as np

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
        success_labels.append(labels[success_idxs])

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