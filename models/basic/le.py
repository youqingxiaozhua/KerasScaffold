from tensorflow import keras
from tensorflow.keras import layers as l
from absl import flags
FLAGS = flags.FLAGS

__all__ = ['lenet', 'lenet_bn']


def lenet(input_shape, classes, *args, **kwargs):
    inputs = keras.Input(shape=input_shape, name='digits')
    x = l.Conv2D(64, (3, 3), padding='same', activation='relu')(inputs)
    x = l.MaxPool2D()(x)

    x = l.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = l.MaxPool2D()(x)

    x = l.Flatten()(x)
    x = l.Dense(128, activation='relu')(x)

    outputs = l.Dense(classes, activation='softmax', name='predictions')(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


def lenet_bn():
    inputs = keras.Input(shape=FLAGS.input_shape, name='digits')
    x = l.Conv2D(64, (3, 3), padding='same')(inputs)
    x = l.BatchNormalization()(x)
    x = l.ReLU()(x)
    x = l.MaxPool2D()(x)

    x = l.Conv2D(128, (3, 3), padding='same')(x)
    x = l.BatchNormalization()(x)
    x = l.ReLU()(x)
    x = l.MaxPool2D()(x)

    x = l.Flatten()(x)
    x = l.Dense(128)(x)
    x = l.BatchNormalization()(x)
    x = l.ReLU()(x)

    outputs = l.Dense(FLAGS.class_num, activation='softmax', name='predictions')(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

