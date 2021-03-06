import tensorflow as tf
import foolbox.attacks as fa
import numpy as np
import eagerpy as ep

from foolbox import TensorFlowModel, accuracy, samples
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.losses import categorical_crossentropy, sparse_categorical_crossentropy
#from tensorflow.math import reduce_mean, reduce_sum, square, exp, log, add, subtract, multiply, argmax, count_nonzero
from math import ceil

from foolbox.attacks import LinfProjectedGradientDescentAttack, LinfFastGradientAttack, LinfBasicIterativeAttack, L2BasicIterativeAttack, L2ProjectedGradientDescentAttack
from foolbox.models.base import Model as FModel
from typing import Optional, Callable
from eagerpy.tensor import TensorFlowTensor

class CustomLossLinfPGDAttack(LinfProjectedGradientDescentAttack):
    def __init__(self, loss_fn, steps, rel_stepsize, random_start, **kwargs):
        super().__init__(**kwargs)
        self.loss_fn = loss_fn
        self.steps = steps
        self.rel_stepsize = rel_stepsize
        self.random_start = random_start

    def get_loss_fn(self, model: FModel, labels: ep.Tensor) -> Callable[[ep.Tensor], ep.Tensor]:
        def loss_fn(inputs: ep.Tensor) -> ep.Tensor:
            logits = model(inputs)
            loss = self.loss_fn(labels.raw, logits.raw)
            return TensorFlowTensor(tf.reduce_sum(loss))
            #return self.loss_fn(labels.raw, logits.raw)
        return loss_fn

class CustomLossL2PGDAttack(L2ProjectedGradientDescentAttack):
    def __init__(self, loss_fn, steps, rel_stepsize, random_start, **kwargs):
        super().__init__(**kwargs)
        self.loss_fn = loss_fn
        self.steps = steps
        self.rel_stepsize = rel_stepsize
        self.random_start = random_start

    def get_loss_fn(self, model: FModel, labels: ep.Tensor) -> Callable[[ep.Tensor], ep.Tensor]:
        def loss_fn(inputs: ep.Tensor) -> ep.Tensor:
            logits = model(inputs)
            loss = self.loss_fn(labels.raw, logits.raw)
            return TensorFlowTensor(tf.reduce_sum(loss))
            #return self.loss_fn(labels.raw, logits.raw)
        return loss_fn

class CustomLossLinfFGSMAttack(LinfFastGradientAttack):
    def __init__(self, loss_fn, **kwargs):
        super().__init__(**kwargs)
        self.loss_fn = loss_fn

    def get_loss_fn(self, model: FModel, labels: ep.Tensor) -> Callable[[ep.Tensor], ep.Tensor]:
        def loss_fn(inputs: ep.Tensor) -> ep.Tensor:
            logits = model(inputs)
            loss = self.loss_fn(labels.raw, logits.raw)
            return TensorFlowTensor(tf.reduce_sum(loss))
            #return self.loss_fn(labels.raw, logits.raw)
        return loss_fn

class CustomLossLinfBasicIterativeAttack(LinfBasicIterativeAttack):
    def __init__(self, loss_fn, steps, rel_stepsize, random_start, **kwargs):
        super().__init__(**kwargs)
        self.loss_fn = loss_fn
        self.steps = steps
        self.rel_stepsize = rel_stepsize
        self.random_start = random_start

    def get_loss_fn(self, model: FModel, labels: ep.Tensor) -> Callable[[ep.Tensor], ep.Tensor]:
        def loss_fn(inputs: ep.Tensor) -> ep.Tensor:
            logits = model(inputs)
            loss = self.loss_fn(labels.raw, logits.raw)
            return TensorFlowTensor(tf.reduce_sum(loss))
            #return self.loss_fn(labels.raw, logits.raw)
        return loss_fn

class CustomLossL2BasicIterativeAttack(L2BasicIterativeAttack):
    def __init__(self, loss_fn, steps, rel_stepsize, random_start, **kwargs):
        super().__init__(**kwargs)
        self.loss_fn = loss_fn
        self.steps = steps
        self.rel_stepsize = rel_stepsize
        self.random_start = random_start

    def get_loss_fn(self, model: FModel, labels: ep.Tensor) -> Callable[[ep.Tensor], ep.Tensor]:
        def loss_fn(inputs: ep.Tensor) -> ep.Tensor:
            logits = model(inputs)
            loss = self.loss_fn(labels.raw, logits.raw)
            return TensorFlowTensor(tf.reduce_sum(loss))
            #return self.loss_fn(labels.raw, logits.raw)
        return loss_fn



def categorical_crossentropy_with_logits(y_true, y_pred):
    return categorical_crossentropy(y_true, y_pred, from_logits=True)

def sparse_categorical_crossentropy_with_logits(y_true, y_pred):
    return sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)



def base(attack, model, images, labels, batch_size, epsilons, bounds):
    # Preprocess test data to feed to Foolbox
    labels = np.argmax(labels, axis=1) # From categorical to raw labels
    images, labels = tf.convert_to_tensor(images, dtype_hint=tf.float32), tf.convert_to_tensor(labels, dtype_hint=tf.int64)
    fmodel = TensorFlowModel(model, bounds=bounds)

    outcomes = {}
    for eps in epsilons: outcomes[eps] = (0, 0)

    all_imgs, success_imgs, robustnesses, num_adversarial = [], [], [], []
    batch_size = batch_size if batch_size is not None else len(images)
    num_batches = ceil(len(images) / batch_size)

    for i in range(num_batches):
        last = i == num_batches - 1
        batch_images = images[i*batch_size:(i+1)*batch_size] if not last else images[i*batch_size:]
        batch_labels = labels[i*batch_size:(i+1)*batch_size] if not last else labels[i*batch_size:]

        _, imgs, successes = attack(fmodel, batch_images, batch_labels, epsilons=epsilons)
        successes = successes.numpy()

        num_attacks = len(batch_images)

        for j in range(len(epsilons)):
            success_idxs = successes[j] == 1

            try:
                all_imgs[j] = np.append(all_imgs[j], imgs[j], axis=0)
                success_imgs[j] = np.append(success_imgs[j], imgs[j][success_idxs], axis=0)
            except:
                all_imgs.append(imgs[j])
                success_imgs.append(imgs[j][success_idxs])

            eps = epsilons[j]
            num_successes = np.count_nonzero(success_idxs)
            outcome = (num_successes, num_attacks)
            outcome_so_far = outcomes[eps]
            outcomes[eps] = tuple(map(sum, zip(outcome, outcome_so_far)))

    for i, eps in enumerate(epsilons):
        num_successes, num_attacks = outcomes[eps]

        if len(success_imgs[i]) > 0:
            preds = np.argmax(model.predict(success_imgs[i]), axis=1)
            predicted_advs = np.count_nonzero(preds == 10)
            num_adversarial.append(predicted_advs)
            num_successes -= predicted_advs

        robustness = round(1.0 - num_successes / num_attacks, 3)
        robustnesses.append(robustness)

        print("For epsilon = {}, there were {}/{} successful attacks (robustness = {})".format(eps, num_successes, num_attacks, round(1.0 - num_successes / num_attacks, 3)))
    
    return all_imgs, success_imgs, robustnesses, num_adversarial

def pgd_attack(model, images, labels, norm="linf", loss_fn=sparse_categorical_crossentropy_with_logits, steps=40, rel_stepsize=0.01/0.3, random_start=True, batch_size=None, epsilons=[0.03, 0.1, 0.3], bounds=(0,1)):
    """ Evaluates robustness against an L-infinity PGD attack with random restart and 40 steps.
    Args:
        model : Tensorflow model to evaluate.
        images : Clean images that will be turned into adversarial examples
        labels : Labels of the clean images
        norm : 'L2' or 'Linf'
    """

    print("Performing PGD attack...")
    if norm.lower() == "l2":
        attack = CustomLossL2PGDAttack(loss_fn, steps, rel_stepsize, random_start)
    if norm.lower() == "linf":
        attack = CustomLossLinfPGDAttack(loss_fn, steps, rel_stepsize, random_start)
    #attack = fa.LinfPGD()
    return base(attack, model, images, labels, batch_size, epsilons, bounds)


def fgsm_attack(model, images, labels, loss_fn=sparse_categorical_crossentropy_with_logits, batch_size=None, epsilons=[0.03, 0.1, 0.3], bounds=(0,1)):
    """ Evaluates robustness against an L-infinity FGSM attack without random restart.
    Args:
        model : Tensorflow model to evaluate.
        images : Clean images that will be turned into adversarial examples
        labels : Labels of the clean images
    """

    print("Performing FGSM attack...")
    attack = CustomLossLinfFGSMAttack(loss_fn)
    #attack = fa.FGSM()
    return base(attack, model, images, labels, batch_size, epsilons, bounds)


def basic_iterative_attack(model, images, labels, norm="linf", loss_fn=sparse_categorical_crossentropy_with_logits, steps=10, rel_stepsize=0.2, random_start=False, batch_size=None, epsilons=[0.03, 0.1, 0.3], bounds=(0,1)):
    """ Evaluates robustness against an L-infinity Basic Iterative Attack with 10 steps.
    Args:
        model : Tensorflow model to evaluate.
        images : Clean images that will be turned into adversarial examples
        labels : Labels of the clean images
        norm : "L2" or "Linf"
    """

    print("Performing Basic Iterative Attack...")
    if norm.lower() == "l2":
        attack = CustomLossL2BasicIterativeAttack(loss_fn, steps, rel_stepsize, random_start)
    if norm.lower() == "linf":
        attack = CustomLossLinfBasicIterativeAttack(loss_fn, steps, rel_stepsize, random_start)
    #attack = fa.LinfBasicIterativeAttack()
    return base(attack, model, images, labels, batch_size, epsilons, bounds)


def cw_attack(model, images, labels, batch_size=None, epsilons=[0.03, 0.1, 0.3], bounds=(0,1)):
    """ Evaluates robustness against an L2 CW attack .
    Args:
        model : Tensorflow model to evaluate.
        images : Clean images that will be turned into adversarial examples
        labels : Labels of the clean images
    """

    print("Performing CW Attack...")
    attack = fa.L2CarliniWagnerAttack()
    return base(attack, model, images, labels, batch_size, epsilons, bounds)
    

def cvae_pgd(model, images, labels, loss_fn=categorical_crossentropy, epsilon=0.3, batch_size=None, training=False):
    """ 
        Returns adversarial examples (created from x) and robustness (#unsuccessful attacks / #attacks).
        Uses 40 steps and random restart.

        Args:
            model : CVAE model that should output [z_mean, z_log_var, reconstructions, outputs].
            training : Set to True if using the method for adversarial training.
    """

    steps = 40
    rel_stepsize = 0.01/0.3
    stepsize = rel_stepsize * epsilon
    model_lb, model_ub = 0, 1 # model bounds
    
    def get_random_start(x):
        return x + tf.random.uniform(x.shape, -epsilon, epsilon)

    def project(x_adv, x):
        return x + tf.clip_by_value(x_adv - x, -epsilon, epsilon)

    def gradients(x, y):
        with tf.GradientTape() as tape:
            tape.watch(x)
            if training:
                _, _, _, preds = model([x,y])
            else:
                preds = model(x)
            loss = loss_fn(y, preds)
        gradients = tape.gradient(loss, x)
        return gradients

    def pgd(model, x, y, batch_size):
        print("Performing PGD attack...")
        num_correct_preds = 0
        num_examples = x.shape[0]
        batch_size = batch_size if batch_size is not None else num_examples
        num_batches = ceil(num_examples / batch_size)
        
        for i in range(num_batches):
            last = i == num_batches - 1
            batch_images = x[i*batch_size:(i+1)*batch_size] if not last else x[i*batch_size:]
            batch_labels = y[i*batch_size:(i+1)*batch_size] if not last else y[i*batch_size:]

            batch_advs = get_random_start(batch_images)
            batch_advs = tf.clip_by_value(batch_advs, model_lb, model_ub)
            for _ in range(steps): 
                grads = gradients(batch_advs, batch_labels)
                sign_grads = tf.sign(grads)
                batch_advs = batch_advs + stepsize * sign_grads
                batch_advs = project(batch_advs, batch_images)
                batch_advs = tf.clip_by_value(batch_advs, model_lb, model_ub) 
            # Add to num_correct_preds
            if training:
                _, _, _, preds = model([batch_advs, batch_labels])
            else:
                preds = model(batch_advs)
            y_preds = argmax(preds, axis=1)
            y_true = argmax(batch_labels, axis=1)
            num_correct_preds += np.count_nonzero(y_preds == y_true)

        robustness = num_correct_preds / num_examples
        print("For epsilon = {}, there were {}/{} successful attacks (robustness = {})".format(epsilon, num_examples - num_correct_preds, num_examples, robustness))

        return batch_advs, robustness

    return pgd(model, images, labels, batch_size)