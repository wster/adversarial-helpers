from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.losses import sparse_categorical_crossentropy

def categorical_crossentropy_with_logits(y_true, y_pred):
    return categorical_crossentropy(y_true, y_pred, from_logits=True)

def sparse_categorical_crossentropy_with_logits(y_true, y_pred):
    return sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)