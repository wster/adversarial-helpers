import tensorflow as tf
from foolbox import TensorFlowModel, accuracy, samples
import foolbox.attacks as fa

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
            y = tf.reshape(y, shape=(1,))
            # Replace clean example with adversarial example for adversarial training
            x_adv, _, _ = fa.LinfPGD()(fmodel, x, y, epsilons=[0.3])
            train_step(x_adv, y)
            progress_bar_train.add(x.shape[0], values=[('loss', train_loss.result())])

    return model

