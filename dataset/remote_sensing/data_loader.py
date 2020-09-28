import os
import random

import tensorflow as tf
import tensorflow_io as tfio
import numpy as np
from absl import flags, logging
import glob
from PIL import Image
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'sans-serif']


from utils.ds_preprocess import write_items_to_txt
from utils.evaluate import evaluate_batch
from utils.flags import define_flages
from utils.path import BASE_DIR

FLAGES = flags.FLAGS

AUTOTUNE = tf.data.experimental.AUTOTUNE
ds_dir = os.path.join(BASE_DIR, 'dataset/remote_sensing')


class DataLoader:
    def __init__(self, batch_size, *args, **kwargs):
        self.train_size, self.valid_size, self.test_size = 0, 0, 0
        self.classes = 8
        self.input_shape = [256, 256, 3]
        self.batch_size = batch_size
        self.images = None  # file path
        self.masks = None   # file path
        self.labels = ('Water', 'Transportation', 'Building', 'Arable Land', 'Grass', 'Forest', 'Bare Soil', 'Others')
        #                                                         耕地            草地                 裸土
        if FLAGES.flag_values_dict():
            FLAGES.set_default('classes', self.classes)
            FLAGES.set_default('input_shape', self.input_shape)

    def get_filelist(self, mode, type):
        assert type in {'image', 'mask'}
        with open(os.path.join(ds_dir, '%s.txt' % mode), 'r') as f:
            images = f.read().split('\n')
        if type == 'image':
            return images
        masks = [os.path.join(ds_dir, 'RemoteSensing/train/label/%s.png'%os.path.basename(i)[:-4]) for i in images]
        return masks

    def load_mask(self, file):
        mask = Image.open(file.numpy())
        mask = np.array(mask)
        mask = mask[:, :, np.newaxis]
        mask = mask / 100 - 1
        return mask

    def load(self, image, mask=None):
        image = tf.io.read_file(image)
        # image = tf.image.decode_image(image)  # can not decode tiff
        image = tfio.experimental.image.decode_tiff(image)
        image = image[:,:,:3] # tiff have 4 channels: RGBA, all A channel is 255 here
        image = tf.cast(image, tf.float16)
        image /= 255.
        if mask is not None:
            mask = tf.py_function(self.load_mask, [mask], tf.int32)
            return image, mask
        else:
            return image

    def augment(self, image, mask):
        if random.random() > 0.5:
            image = tf.image.flip_left_right(image)
            mask = tf.image.flip_left_right(mask)
        # if random.random() > 0.66:
        #     image = tf.cast(image, tf.int32)
        #     stacked = tf.image.random_crop(tf.stack([image, mask], axis=0), size=(2, 200, 200, 3))
        #     stacked = tf.image.resize(stacked, self.input_shape[:2], method='nearest')
        #     image, mask = stacked[0], stacked[1]
        # 随机旋转
        rotate = random.choice((0, 1, 2, 3))
        image = tf.image.rot90(image, k=rotate)
        mask = tf.image.rot90(mask, k=rotate)

        image = tf.image.random_brightness(image, max_delta=0.2)
        image = tf.image.random_contrast(image, 0.2, 0.5)

        return image, mask

    def one_hot(self, image, mask):
        mask = tf.squeeze(mask)
        # ont hot by hand
        mask = tf.one_hot(mask, depth=self.classes)
        mask = tf.cast(mask, tf.float16)

        image = tf.reshape(image, self.input_shape)
        mask = tf.reshape(mask, (self.input_shape[0], self.input_shape[1], self.classes))
        return image, mask

    def get(self, mode):
        with open(os.path.join(ds_dir, '%s.txt' % mode), 'r') as f:
            images = f.read().split('\n')
        setattr(self, '%s_size' % mode, len(images))
        image_ds = tf.data.Dataset.from_tensor_slices(images)
        if mode == 'predict':
            ds = image_ds.map(self.load, num_parallel_calls=AUTOTUNE)
        else:
            masks = [os.path.join(ds_dir, 'RemoteSensing/train/label/%s.png'%os.path.basename(i)[:-4]) for i in images]
            mask_ds = tf.data.Dataset.from_tensor_slices(masks)
            ds = tf.data.Dataset.zip((image_ds, mask_ds))
            ds = ds.map(self.load, num_parallel_calls=AUTOTUNE)
            ds.cache()
            if 'train' in mode:
                tf.print('** Augment Start **')
                ds = ds.map(self.augment, num_parallel_calls=AUTOTUNE)
                tf.print('** Augment End **')
                ds = ds.shuffle(buffer_size=1024)
            ds = ds.map(self.one_hot, num_parallel_calls=AUTOTUNE)
        tf.print(ds)
        ds = ds.batch(self.batch_size)
        ds = ds.prefetch(buffer_size=AUTOTUNE)
        return ds

    def visulize(self, mode):
        ds = self.get(mode)
        image_batch, label_batch = next(iter(ds))
        tf.print('image shape:', image_batch.shape)
        tf.print('label shape:', label_batch.shape)
        m = self.batch_size
        plt.figure()
        for i in range(self.batch_size):
                plt.subplot(m, 2, i * 2 + 1)
                plt.imshow(tf.squeeze(image_batch[i]))
                plt.xticks([])
                plt.yticks([])
                plt.subplot(m, 2, i * 2 + 2)
                plt.imshow(tf.squeeze(tf.argmax(label_batch[i], axis=-1)))
                plt.xticks([])
                plt.yticks([])
                # plt.xlabel()
        plt.show()

    def color_map_viz(self):
        row_size = 300
        col_size = 500
        cmap = [i for i in range(self.classes)]
        array = np.empty((row_size*self.classes, col_size), dtype='int8')
        for i in range(self.classes):
            array[i*row_size:i*row_size+row_size] = cmap[i]

        plt.figure()
        plt.imshow(array)
        plt.yticks([row_size * i + row_size / 2 for i in range(self.classes)], self.labels)
        plt.xticks([])
        plt.show()

    def visualize_evaluate(self, model, mode):
        images = self.get_filelist(mode, 'image')
        masks = self.get_filelist(mode, 'mask')
        evaluate_batch(model, self, images, masks, self.classes, plot_num=3, skip=0)

    def cal_freq(self):
        result = {
            'total': 0,
        }
        ds = self.get('train')
        for _, mask in iter(ds):
            result['total'] += self.batch_size * self.input_shape[0] * self.input_shape[1]
            y, idx, count = tf.unique_with_counts(tf.reshape(mask, (-1, )))
            y = y.numpy()
            count = count.numpy()
            for i in range(len(y)):
                if y[i] in result:
                    result[y[i]] += count[i]
                else:
                    result[y[i]] = 0
        for k, v in result.items():
            print(k, v, v/result['total'])



if __name__ == '__main__':
    define_flages()
    # make list
    # image_path = os.path.join(ds_dir, 'RemoteSensing/train/image')
    # all_files = glob.glob(image_path, '*.tif')
    # print(len(all_files))
    # with open(os.path.join(ds_dir, 'train_val.txt'), 'w') as f:
    #     f.write('\n'.join(all_files))
    # random.shuffle(all_files)
    # write_items_to_txt(os.path.join(ds_dir, 'train.txt'), all_files[:-5000])
    # write_items_to_txt(os.path.join(ds_dir, 'valid.txt'), all_files[-5000:])
    # test_files = glob.glob('%s/*.tif' % os.path.join(ds_dir, 'RemoteSensing/image_A'))
    # print('len of test:', len(test_files))
    # write_items_to_txt(os.path.join(ds_dir, 'test.txt'), test_files)



    # test dataset
    ds = DataLoader(batch_size=1024)
    # ds.visulize('train')
    ds.cal_freq()



