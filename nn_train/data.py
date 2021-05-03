import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


def load_data(dataset_url, image_size, batch_size):
    """ Load mask images with augmentations """

    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=20,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2
    )

    train_generator = train_datagen.flow_from_directory(
        dataset_url,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True,
        subset='training',
    )
    val_generator = train_datagen.flow_from_directory(
        dataset_url,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True,
        subset='validation',
    )

    return train_generator, val_generator


def show_examples(image_generator):
    class_names = ['Mask', 'No mask']

    plt.figure(figsize=(10, 10))
    images, labels = image_generator.next()
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i] / 255)
        plt.title(class_names[np.argmax(labels[i])])
        plt.axis("off")
