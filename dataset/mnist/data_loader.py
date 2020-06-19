import tensorflow as tf
import numpy as np
from absl import flags, logging
FLAGES = flags.FLAGS

AUTOTUNE = tf.data.experimental.AUTOTUNE


class DataLoader:
    def __init__(self, *args, **kwargs):
        self.train_size, self.valid_size, self.test_size = 0, 0, 0
        FLAGES.set_default('classes', 10)
        FLAGES.set_default('input_shape', [28, 28, 1])

    def pre_process(self, data_pair):
        x, y = data_pair
        x = x/255.0
        x = x[:, :, :, np.newaxis]
        y = tf.one_hot(y, FLAGES.classes)
        return x, y

    def get(self, mode):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

        x_train, x_val = x_train[:-10000], x_train[-10000:]
        y_train, y_val = y_train[:-10000], y_train[-10000:]
        data = {
            'train': (x_train, y_train),
            'valid': (x_val, y_val),
            'test': (x_test, y_test)
        }
        x, y = data[mode]

        setattr(self, '%s_size'%mode, len(x))
        logging.info('%s_x shape: %s' % (mode, tf.shape(x)))
        x, y = self.pre_process((x, y))
        ds = tf.data.Dataset.from_tensor_slices((x, y))
        ds = ds.shuffle(len(x))
        # if mode == 'train':
        #     ds.repeat()
        ds = ds.batch(FLAGES.batch_size)

        ds = ds.prefetch(AUTOTUNE)
        return ds

