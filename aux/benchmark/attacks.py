import tensorflow as tf
import foolbox.attacks as fa
import numpy as np
import sys

from foolbox import TensorFlowModel, accuracy, samples
from tensorflow.keras.utils import to_categorical
from math import ceil

def base(attack, model, images, labels, batch_size, epsilons, bounds):
    # Preprocess test data to feed to Foolbox
    labels = np.argmax(labels, axis=1) # From categorical to raw labels
    images, labels = tf.convert_to_tensor(images, dtype_hint=tf.float32), tf.convert_to_tensor(labels, dtype_hint=tf.int64)
    fmodel = TensorFlowModel(model, bounds=bounds)

    print("Clean accuracy:", model.evaluate(images, labels)[1])
    print("")

    outcomes = {}
    for eps in epsilons: outcomes[eps] = (0, 0)

    success_imgs, success_labels = [], []
    batch_size = batch_size if batch_size is not None else len(images)
    num_batches = ceil(len(images) / batch_size)

    print_progress(0, num_batches)

    for i in range(num_batches):
        last = i == num_batches - 1
        batch_images = images[i*batch_size:(i+1)*batch_size] if not last else images[i*batch_size:]
        batch_labels = labels[i*batch_size:(i+1)*batch_size] if not last else labels[i*batch_size:]

        # Report robustness accuracy
        _, imgs, successes = attack(fmodel, batch_images, batch_labels, epsilons=epsilons)
        successes = successes.numpy()

        num_attacks = len(batch_images)

        for i in range(len(epsilons)):
            success_idxs = successes[i] == 1

            success_imgs.append(imgs[i][success_idxs])
            categorical_labels = to_categorical(batch_labels.numpy()[success_idxs])
            success_labels.append(categorical_labels)

            eps = epsilons[i]
            num_successes = np.count_nonzero(success_idxs)
            outcome = (num_successes, num_attacks)
            outcome_so_far = outcomes[eps]
            outcomes[eps] = tuple(map(sum, zip(outcome, outcome_so_far)))
        
        print_progress(i, num_batches)

    for eps in epsilons:
        num_successes, num_attacks = outcomes[eps]
        print("For epsilon = {}, there were {}/{} successful attacks (robustness = {})".format(epsilons[i], num_successes, num_attacks, round(1.0 - num_successes / num_attacks, 2)))
    
    return success_imgs, success_labels 

def pgd_attack(model, images, labels, batch_size=None, epsilons=[0.03, 0.1, 0.3], bounds=(0,1)):
    """ Evaluates robustness against an L-infinity PGD attack with random restart and 40 steps.
    Args:
        model : Tensorflow model to evaluate.
        images : Clean images that will be turned into adversarial examples
        labels : Labels of the clean images
    """

    print("Performing PGD attack...")
    attack = fa.LinfPGD()
    return base(attack, model, images, labels, batch_size, epsilons, bounds)


def fgsm_attack(model, images, labels, batch_size=None, epsilons=[0.03, 0.1, 0.3], bounds=(0,1)):
    """ Evaluates robustness against an L-infinity FGSM attack without random restart.
    Args:
        model : Tensorflow model to evaluate.
        images : Clean images that will be turned into adversarial examples
        labels : Labels of the clean images
    """

    print("Performing FGSM attack...")
    attack = fa.FGSM()
    return base(attack, model, images, labels, batch_size, epsilons, bounds)


def basic_iterative_attack(model, images, labels, batch_size=None, epsilons=[0.03, 0.1, 0.3], bounds=(0,1)):
    """ Evaluates robustness against an L-infinity Basic Iterative Attack with 10 steps.
    Args:
        model : Tensorflow model to evaluate.
        images : Clean images that will be turned into adversarial examples
        labels : Labels of the clean images
    """

    print("Performing Basic Iterative Attack...")
    attack = fa.LinfBasicIterativeAttack()
    return base(attack, model, images, labels, batch_size, epsilons, bounds)

def print_progress(iteration, total, prefix='', suffix='', decimals=1, bar_length=100):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        bar_length  - Optional  : character length of bar (Int)
    """
    str_format = "{0:." + str(decimals) + "f}"
    percents = str_format.format(100 * (iteration / float(total)))
    filled_length = int(round(bar_length * iteration / float(total)))
    bar = '█' * filled_length + '-' * (bar_length - filled_length)

    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),

    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()