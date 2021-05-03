import tensorflow as tf
import math
import matplotlib.pyplot as plt


class BaseModel(tf.keras.layers.Layer):
    """ MobileNetV2 without head """

    def __init__(self, image_shape):
        super(BaseModel, self).__init__()

        self.base_net = tf.keras.applications.MobileNetV2(
            input_shape=image_shape,
            include_top=False,
            weights='imagenet'
        )

        self.base_net.trainable = False

    def call(self, inputs):
        return self.base_net(inputs)

    def fineTune(self, layers_per):
        layers = math.floor(layers_per * len(self.base_net.layers))
        print(layers)

        self.base_net.trainable = True
        for layer in self.base_net.layers[:layers]:
            layer.trainable = False


class ClassificationHead(tf.keras.layers.Layer):
    """ Flatten input and classify """

    def __init__(self):
        super(ClassificationHead, self).__init__()

        self.pooling = tf.keras.layers.GlobalAveragePooling2D()
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dropout = tf.keras.layers.Dropout(0.5)
        self.dense2 = tf.keras.layers.Dense(2)

    def call(self, inputs):
        x = self.pooling(inputs)
        x = self.dense1(x)
        x = self.dropout(x)
        return self.dense2(x)


class PreprocessInput(tf.keras.layers.Layer):
    """ Preprocess inputs to fit MobileNetV2 """

    def __init__(self):
        super(PreprocessInput, self).__init__()

        self.preprocess = tf.keras.applications.mobilenet_v2.preprocess_input

    def call(self, inputs):
        return self.preprocess(inputs)


class Model(tf.keras.Model):
    """ Complete pipeline for MobileNetV2 based nn """

    def __init__(self, image_shape):
        super(Model, self).__init__()

        self.preprocess = PreprocessInput()
        self.base = BaseModel(image_shape)
        self.head = ClassificationHead()

    def call(self, inputs):
        x = self.preprocess(inputs)
        x = self.base(x)
        return self.head(x)

    def fineTune(self, layers_per):
        self.base.fineTune(layers_per)


def plot_curves(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()), 1])
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.ylim([0, 1.0])
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.show()


def plot_curves_ft(history, history_fine, epochs):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    acc += history.history['accuracy']
    val_acc += history.history['val_accuracy']

    loss += history.history['loss']
    val_loss += history.history['val_loss']

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.ylim([0.8, 1])
    plt.plot([epochs - 1, epochs - 1],
             plt.ylim(), label='Start Fine Tuning')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.ylim([0, 1.0])
    plt.plot([epochs - 1, epochs - 1],
             plt.ylim(), label='Start Fine Tuning')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.show()
