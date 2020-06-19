import os
import shutil
import logging

import random
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from tqdm import tqdm
from utils.path import DATASET_DIR

AUTOTUNE = tf.data.experimental.AUTOTUNE


class Process:
    def __init__(self, source, dest, split=(8, 1, 1), clean=False, segment=False,
                 label_source='',
                 ):
        """
        :param source: 原数据文件夹，每个子目录为一类
        :param dest: 目标文件夹
        :param split: train, val, test划分比例
        :param clean: 是否删除原文件夹重新开始
        :param segment: 是否是分割任务
        :param label_source: 分割的label所在文件夹
        """
        self.source = source
        self.dest = dest
        self.jpg_path = os.path.join(dest, 'JPEGImages')
        self.split = split
        self.segment = segment
        if self.segment:
            self.label_source = label_source
            self.png_path = os.path.join(dest, 'SegmentationClassRaw')

        self.samples = []  # c个子list分别存储每类的文件名
        self.train = []
        self.val = []
        self.test = []

        if clean:
            shutil.rmtree(dest)
        os.makedirs(self.jpg_path, exist_ok=True)
        if self.segment:
            os.makedirs(self.png_path, exist_ok=True)

    def _list_samples(self):
        # 列出所有filename和标签
        # 带有文件夹名称，如 ./Cat/1.jpg -> Cat-1.jpg
        self.class_paths = next(os.walk(self.source))[1]
        for c in self.class_paths:
            self.samples.append(
                ['%s/%s' % (c, i) for i in
                 next(os.walk(os.path.join(self.source, c)))[2]
                 ]
            )

    def convert_img_label(self, file, img: Image, label=None):
        """
        处理图片与label
        :param file: filename without extension
        """
        if self.segment:
            return img, label
        return img

    def write_txt(self):
        names = ('train', 'valid', 'test')
        available_samples = set()  # self.samples中有一些没有分割标注
        for i, v in enumerate((self.train, self.val, self.test)):
            available_samples = available_samples | set(v)
            content = '\n'.join(v)
            with open(os.path.join(self.dest, '%s.csv' % names[i]), 'w') as r:
                r.write(content)

        with open(os.path.join(self.dest, 'all.csv'), 'w') as r:
            r.write('\n'.join(available_samples))

    def process(self, label_index, filename):
        """处理单个sample
        :param label_index: 图片label所在文件夹名称
        :param filename: 当前sample文件名
        :return False if there is no label, run function will ignore this sample
        """
        img_path = os.path.join(self.source, filename)
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            logging.error('read image{} error: {}'.format(img_path, e))
            return False
        file = filename[:-4]
        if self.segment:
            label_name = file + '.png'
            label_path = os.path.join(self.label_source, label_name)
            # label = cv.imread(label_path, cv.IMREAD_GRAYSCALE)
            try:
                label = Image.open(label_path)
            except FileNotFoundError:
                return
            im, label = self.convert_img_label(file, img, label)
            if label is None:
                return
            label.save(os.path.join(self.png_path, label_name), 'PNG')
        else:
            im = self.convert_img_label(file, img)
        # 将filename中的目录/改为-（防止重名）
        im.save(os.path.join(self.jpg_path, filename.replace('/', '-')), 'JPEG')
        return True

    def get_set_txt(self, filename, label):
        """根据是否是分割任务返回Image Set的csv文件中的一行"""
        filename = filename.replace('/', '-')
        if self.segment:
            return '%s' % filename[:-4]
        else:
            return '%s,%s' % (filename, label)

    def run(self):
        self._list_samples()
        # 处理图片并写入目标目录
        for label, c in enumerate(self.samples):
            total_num = len(c)
            for count, file in tqdm(enumerate(c)):
                if not self.process(label, file):
                    continue
                rate = (count + 1) / total_num
                txt = self.get_set_txt(file, label)
                if rate <= self.split[0] / 10:
                    self.train.append(txt)
                elif rate <= (self.split[0] + self.split[1]) / 10:
                    self.val.append(txt)
                else:
                    self.test.append(txt)

        # 写入txt
        self.write_txt()

        logging.info('process done')
        a, b = (self.train, self.val, self.test), ('train', 'valid', 'test')
        for i in range(3):
            logging.info('%s size: %s' % (b[i], len(a[i])))


class DataSet:
    def __init__(self, name, input_shape, num_classes, batch_size):
        self._name = name
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.batch_size = batch_size

        self.train_size = 0
        self.ds_dir = os.path.join(DATASET_DIR, self._name, 'processed')
        self.image_dir = os.path.join(self.ds_dir, 'JPEGImages')

    def read_image(self, path):
        path = self.image_dir + '/' + tf.strings.split(path, '/')[-1]
        image = tf.io.read_file(path)
        image = tf.cond(
            tf.image.is_jpeg(image),
            lambda: tf.image.decode_jpeg(image, channels=self.input_shape[-1]),
            lambda: tf.image.decode_png(image, channels=self.input_shape[-1]))
        # image = tf.image.resize_with_crop_or_pad(image, self.input_shape[0], self.input_shape[1])
        # resize and cast to float will be done in augment
        image = tf.image.resize(image, (self.input_shape[0:2]))
        image /= 255.
        return image


class ClassifyDataset(DataSet):
    def __init__(self, name, input_shape, classes, batch_size=5, *args, **kwargs):
        super().__init__(name, input_shape, classes, batch_size)
        self.image_set_dir = os.path.join(self.ds_dir, 'ImageSets', 'Classification')

    def preprocess(self, image, label):
        """归一化、统一文件大小"""
        tf.print(self.input_shape)
        image = tf.image.resize_with_crop_or_pad(image, *self.input_shape[0:2])
        image = tf.image.convert_image_dtype(image, tf.float32)  # Cast and normalize the image to [0,1]
        image = image / 255.0
        return image, label

    def augment(self, image, label):
        return image, label

    def get(self, mode):
        csv_file = os.path.join(self.image_set_dir, '%s.csv' % mode)
        lines = np.loadtxt(csv_file, dtype=np.unicode, delimiter=',')
        setattr(self, '%s_size' % mode, lines.shape[0])
        logging.info('%s size: %s' % (mode, lines.shape[0]))
        x = lines[:, 0]
        y = lines[:, 1].astype(int)
        x_ds = tf.data.Dataset.from_tensor_slices(list(x))
        y_ds = tf.data.Dataset.from_tensor_slices(list(y))

        image_ds = x_ds.map(self.read_image, num_parallel_calls=AUTOTUNE)
        label_ds = y_ds.map(lambda x: tf.one_hot(x, self.num_classes), num_parallel_calls=AUTOTUNE)

        image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))
        ds = image_label_ds.cache()
        # ds = ds.map(self.preprocess, num_parallel_calls=AUTOTUNE)
        ds = image_label_ds.shuffle(buffer_size=20480)
        if mode == 'train':
            ds = ds.map(self.augment, num_parallel_calls=AUTOTUNE)
        ds = ds.batch(self.batch_size)
        # if mode == 'train':
        #     ds = ds.repeat()
        ds = ds.prefetch(buffer_size=AUTOTUNE)
        return ds

    def get_predict(self, filename='test.csv'):
        csv_file = os.path.join(self.image_set_dir, filename)
        lines = np.loadtxt(csv_file, dtype=np.unicode, delimiter=',')
        x_ds = tf.data.Dataset.from_tensor_slices(lines)
        image_ds = x_ds.map(self.read_image, num_parallel_calls=AUTOTUNE)
        ds = image_ds.batch(self.batch_size)
        ds = ds.prefetch(buffer_size=AUTOTUNE)
        return ds

    def plot_image(self, mode):
        train_ds = self.get(mode)
        image_batch, label_batch = next(iter(train_ds))
        tf.print('image shape:', image_batch.shape)
        tf.print('label shape:', label_batch.shape)
        m = self.batch_size // 3 + 1
        plt.figure()
        for i in range(self.batch_size):
            for j in range(3):
                plt.subplot(m, 3, i + 1)
                plt.imshow(tf.squeeze(image_batch[i]), cmap='gray')
                label = tf.argmax(label_batch[i], axis=-1)
                plt.xticks([])
                plt.yticks([])
                plt.xlabel(label.numpy())
        plt.show()


def op_with_random(image, func, rand=0.5, *args, **kwargs):
    """func第一个输入需要是image"""
    if random.random() <= rand:
        return func(image, *args, **kwargs)
    return image


