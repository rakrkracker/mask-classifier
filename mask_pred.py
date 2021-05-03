import tensorflow as tf


def load_model():
    # Load model
    model = tf.keras.models.load_model('model/mask_pred.h5')

    # Return model
    return model
