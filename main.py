import importlib
import os
import sys
from math import ceil

from absl import app
import logging

import tensorflow as tf
from tensorflow.keras.metrics import MeanIoU
from tensorflow.keras.optimizers import SGD
from tensorflow.python.keras.metrics import Recall, Precision

from models import *
from utils.callbacks import model_checkpoint, tensorboard, early_stopping
from utils.flags import define_flages
from utils.path import BASE_DIR

flags = define_flages()
FLAGS = flags.FLAGS


def main(argv):
    del argv
    # path
    data_dir = os.path.join(BASE_DIR, 'dataset', FLAGS.dataset)
    exp_dir = os.path.join(data_dir, 'exp', FLAGS.exp_name)
    model_dir = os.path.join(exp_dir, 'ckpt')
    log_dir = exp_dir
    os.makedirs(model_dir, exist_ok=True)
    # os.makedirs(log_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'model-{epoch:04d}.ckpt.h5')

    # logging
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(log_dir, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)
    logging.info('------------------------------experiment start------------------------------------')

    for i in ('exp_name', 'dataset', 'model', 'mode', 'lr',):
        logging.info('%s: %s' % (i, FLAGS.get_flag_value(i, '########VALUE MISSED#########')))
    logging.info(FLAGS.flag_values_dict())

    # resume from checkpoint
    largest_epoch = 0
    resume_file_exist = False
    if FLAGS.resume:
        chkpts = tf.io.gfile.glob(model_dir + '/*.ckpt.h5')
        if len(chkpts):
            largest_epoch = sorted([int(i[-12:-8]) for i in chkpts], reverse=True)[0]
            print('resume from epoch', largest_epoch)
            resume_file_exist = True

    dataset = importlib.import_module('dataset.%s.data_loader' % FLAGS.dataset).DataLoader(**FLAGS.flag_values_dict())
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = globals()[FLAGS.model](**FLAGS.flag_values_dict())
        # model = alexnet()
        if FLAGS.resume and resume_file_exist:
            logging.info('resume from previous ckp: %s' % largest_epoch)
            model.load_weights(model_path.format(epoch=largest_epoch))
        model.compile(
            optimizer=SGD(momentum=0.9),
            loss='binary_crossentropy',
            metrics=["accuracy",
                     Recall(),
                     Precision(),
                     # MeanIoU(num_classes=FLAGS.classes)
                     ],
        )
        model.summary()
        verbose = 1 if FLAGS.debug is True else 2
        if 'train' in FLAGS.mode:
            callbacks = [
                model_checkpoint(filepath=model_path, monitor=FLAGS.model_checkpoint_monitor),
                tensorboard(log_dir=os.path.join(exp_dir, 'tb-logs')),
                early_stopping(patience=FLAGS.early_stopping_patience)
            ]
            train_ds = dataset.get('train')  # get first to calculate train size
            steps_per_epoch = dataset.train_size // FLAGS.batch_size
            model.fit(
                train_ds,
                epochs=FLAGS.epoch,
                validation_data=dataset.get('valid'),
                callbacks=callbacks,
                initial_epoch=largest_epoch,
                verbose=verbose,
                # steps_per_epoch=steps_per_epoch,
            )
        if 'test' in FLAGS.mode:
            # 学习valid
            model.fit(
                dataset.get('valid'),
                epochs=3,
                # callbacks=callbacks,
                verbose=verbose
            )
            model.save_weights(os.path.join(model_dir, 'model.h5'))
            # 测试test
            result = model.evaluate(
                dataset.get('test'),
            )
            logging.info('evaluate result:')
            for i in range(len(result)):
                logging.info('%s:\t\t%s' % (model.metrics_names[i], result[i]))
            # TODO: remove previous checkpoint

# TODO： test的结果如何保存（HParams）


if __name__ == '__main__':
    app.run(main)
