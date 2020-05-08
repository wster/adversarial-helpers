import tensorflow as tf
from foolbox import TensorFlowModel, accuracy, samples
import foolbox.attacks as fa
#from ..benchmark.attacks import cvae_pgd

def adv_fit(model, images, labels, nb_epochs=10, 
            loss_object=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
            optimizer=tf.optimizers.Adam(learning_rate=0.001)): 

    """ Performs adversarial training and returns trained model. Use this instead of Keras "fit" method.
        Trains ONLY on adversarial examples.

    Args:
        model : Tensorflow model to evaluate.
        images : Clean training images that will be used for adversarial training
        labels : Labels of the clean training images
        nb_epochs : Number of training epochs
        loss_object : Tensorflow loss object
        optimizer : Tensorflow optimizer
    """

    # Preprocessing data and setting up Foolbox model
    x_train = tf.convert_to_tensor(images, dtype_hint=tf.float32)
    y_train = tf.convert_to_tensor(labels, dtype_hint=tf.int64)
    fmodel = TensorFlowModel(model, bounds=(0,1))

    # Metric
    train_loss = tf.metrics.Mean(name='train_loss')

    def train_step(x, y):
        with tf.GradientTape() as tape:
            predictions = model(x)
            loss = loss_object(y, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        train_loss(loss)

    # Train model with adversarial training
    for epoch in range(nb_epochs):
        # Display of progress
        progress_bar_train = tf.keras.utils.Progbar(len(x_train))
        for (x, y) in zip(x_train, y_train):
            # Reshape
            x = tf.reshape(x, shape=(1, 28, 28))
            if loss_object is tf.losses.SparseCategoricalCrossentropy(from_logits=True):
                y = tf.reshape(y, shape=(1,))
            else:
                y = tf.reshape(y, shape=(10,))
            # Replace clean example with adversarial example for adversarial training
            print(y)
            x_adv, _, _ = fa.LinfPGD()(fmodel, x, y, epsilons=[0.3])
            train_step(x_adv, y)
            progress_bar_train.add(x.shape[0], values=[('loss', train_loss.result())])

    return model


def cvae_adv_fit(model, images, labels, nb_epochs=10,
                loss_fn=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
                optimizer=tf.optimizers.Adam(learning_rate=0.001),
                batch_size=None):

    """ Performs adversarial training for a CVAE model and returns trained model. Use this instead of Keras "fit" method.
        Trains ONLY on adversarial examples.

    Args:
        model : Tensorflow model to evaluate.
        images : Clean training images that will be used for adversarial training
        labels : Labels of the clean training images
        nb_epochs : Number of training epochs
        loss_fn : Tensorflow loss object
        optimizer : Tensorflow optimizer
        batch_size : Batch size
    """

    # Preprocessing data
    x_train = tf.convert_to_tensor(images, dtype_hint=tf.float32)
    y_train = tf.convert_to_tensor(labels, dtype_hint=tf.int64)

    # Metric
    train_loss = tf.metrics.Mean(name='train_loss')

    def train_step(x, y):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, recons, preds = model([x,y])
            ce_loss = reduce_mean(categorical_crossentropy(y, preds))
            rc_loss = reduce_mean(reduce_sum(mean_squared_error(inputs_x, reconstructions), axis=(1,2)))
            kl_loss = reduce_mean(square(z_mean) + exp(z_log_var) - log(exp(z_log_var) - 1))
            loss = 10*ce_loss + 0.01*rc_loss + 0.1*kl_loss
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        train_loss(loss)

    batch_size = batch_size if batch_size is not None else len(images)
    num_batches = ceil(len(images) / batch_size)

    # Train model with adversarial training
    for epoch in range(nb_epochs):
        print("EPOCH {} BEGINS".format(epoch))
        progress = 0
        for i in range(num_batches):
            last = i == num_batches - 1
            batch_images = x_train[i*batch_size:(i+1)*batch_size] if not last else x_train[i*batch_size:]
            batch_labels = y_train[i*batch_size:(i+1)*batch_size] if not last else y_train[i*batch_size:]

            # Display of progress
            #progress_bar_train = tf.keras.utils.Progbar(len(images))   <-- bugs for some reason
            cur_batch_size = batch_images.shape[0]
            batch_images = tf.reshape(batch_images, shape=(cur_batch_size,28,28,1))
            batch_labels = tf.reshape(batch_labels, shape=(cur_batch_size,10))
            # Replace clean example with adversarial example for adversarial training
            batch_advs, robustness = cvae_pgd(model, batch_images, batch_labels, epsilon=0.3, training=True)
            train_step(batch_advs, batch_labels)
            #progress_bar_train.add(cur_batch_size, values=[('loss', train_loss.result())])
            progress += cur_batch_size
            print("Progress: {} / {}. Robustness of latest batch: {}".format(progress, len(images), robustness))

    return model

