import os

import tensorflow as tf
import matplotlib.pyplot as plt
from absl import flags, app
import logging

from utils.ds_preprocess import Process, ClassifyDataset, op_with_random
from utils.flags import define_flages

FLAGES = flags.FLAGS

AUTOTUNE = tf.data.experimental.AUTOTUNE
BASE_DIR = os.path.dirname(__file__)


class DataLoader(ClassifyDataset):
    def __init__(self, input_shape, classes, batch_size, *args, **kwargs):
        FLAGES.set_default('classes', 2)
        FLAGES.set_default('input_shape', (256, 256, 3))

        super(DataLoader, self).__init__(
            name='cat_dog', **FLAGES.flag_values_dict())
        self.image_set_dir = os.path.join(BASE_DIR, 'processed')

    def augment(self, image, label):
        # image = tf.image.convert_image_dtype(image, tf.float32)  # Cast and normalize the image to [0,1]
        image = tf.image.resize_with_crop_or_pad(image, *self.input_shape[0:2])  # Add 6 pixels of padding
        image = tf.image.random_brightness(image, max_delta=0.5)  # Random brightness
        image = op_with_random(image, tf.image.flip_left_right)
        return image, label


class MakeDataset(Process):
    def __init__(self, *args, **kwargs):
        super().__init__(
            source=os.path.join(BASE_DIR, 'PetImages'),
            dest=os.path.join(BASE_DIR, 'processed'),
            *args, ** kwargs
        )


def test(argv):
    del argv
    flags = define_flages()
    ds = DataLoader(
        input_shape=(256, 256, 3),
        batch_size=2,
        classes=2
    )
    ds.plot_image('train')


if __name__ == '__main__':
    # MakeDataset(clean=True).run()
    app.run(test)
