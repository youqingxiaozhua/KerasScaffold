import importlib
import os
import sys
from math import ceil

from PIL import Image
from absl import app
import logging

import tensorflow as tf
from tensorflow_addons.optimizers import SGDW
from tensorflow.keras.optimizers import SGD
from tensorflow.python.keras.metrics import Recall, Precision
from tensorflow.keras.mixed_precision import experimental as mixed_precision

# enable mixed precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)
print('Compute dtype: %s' % policy.compute_dtype)
print('Variable dtype: %s' % policy.variable_dtype)

from models import *
from losses import *
from metrics import MeanIoU
from utils.callbacks import model_checkpoint, tensorboard, early_stopping, lr_schedule
from utils.ds_preprocess import read_txt, delete_early_ckpt
from utils.evaluate import evaluate_batch
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
    if FLAGS.resume == 'ckpt':
        chkpts = tf.io.gfile.glob(model_dir + '/*.ckpt.h5')
        if len(chkpts):
            largest_epoch = sorted([int(i[-12:-8]) for i in chkpts], reverse=True)[0]
            print('resume from epoch', largest_epoch)
            weight_path = model_path.format(epoch=largest_epoch)
        else:
            weight_path = None
    elif len(FLAGS.resume):
        assert os.path.isfile(FLAGS.resume)
        weight_path = FLAGS.resume
    else:
        weight_path = None

    dataset = importlib.import_module('dataset.%s.data_loader' % FLAGS.dataset).DataLoader(**FLAGS.flag_values_dict())
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = globals()[FLAGS.model](**FLAGS.flag_values_dict())
        # model = alexnet()
        if FLAGS.resume and weight_path:
            logging.info('resume from previous ckp: %s' % largest_epoch)
            model.load_weights(weight_path)
        # model.layers[1].trainable = False
        loss = globals()[FLAGS.loss]
        model.compile(
            optimizer=SGDW(momentum=0.9, learning_rate=FLAGS.lr, weight_decay=FLAGS.weight_decay),
            loss=loss,
            metrics=["accuracy",
                     Recall(),
                     Precision(),
                     MeanIoU(num_classes=FLAGS.classes)
                     ],
        )
        # if 'train' in FLAGS.mode:
        #     model.summary()
        logging.info('There are %s layers in model' % len(model.layers))
        if FLAGS.freeze_layers > 0:
            logging.info('Freeze first %s layers' % FLAGS.freeze_layers)
            for i in model.layers[:FLAGS.freeze_layers]:
                i.trainable = False
        verbose = 1 if FLAGS.debug is True else 2
        if 'train' in FLAGS.mode:
            callbacks = [
                model_checkpoint(filepath=model_path, monitor=FLAGS.model_checkpoint_monitor),
                tensorboard(log_dir=os.path.join(exp_dir, 'tb-logs')),
                early_stopping(monitor=FLAGS.model_checkpoint_monitor, patience=FLAGS.early_stopping_patience),
                lr_schedule(name=FLAGS.lr_schedule, epochs=FLAGS.epoch)
            ]
            file_writer = tf.summary.create_file_writer(os.path.join(exp_dir, 'tb-logs', "metrics"))
            file_writer.set_as_default()
            train_ds = dataset.get('train')  # get first to calculate train size
            model.fit(
                train_ds,
                epochs=FLAGS.epoch,
                validation_data=dataset.get('valid'),
                callbacks=callbacks,
                initial_epoch=largest_epoch,
                verbose=verbose,
            )

            # evaluate before train on valid
            # result = model.evaluate(
            #     dataset.get('test'),
            # )
            # logging.info('evaluate before train on valid result:')
            # for i in range(len(result)):
            #     logging.info('%s:\t\t%s' % (model.metrics_names[i], result[i]))
        if 'test' in FLAGS.mode:
            # 学习valid
            # model.fit(
            #     dataset.get('valid'),
            #     epochs=3,
            #     # callbacks=callbacks,
            #     verbose=verbose
            # )
            # model.save_weights(os.path.join(model_dir, 'model.h5'))
            # 测试test
            result = model.evaluate(
                dataset.get('test'),
            )
            logging.info('evaluate result:')
            for i in range(len(result)):
                logging.info('%s:\t\t%s' % (model.metrics_names[i], result[i]))
            # TODO: remove previous checkpoint
        if 'predict' in FLAGS.mode:
            files = read_txt(os.path.join(BASE_DIR, 'dataset/%s/predict.txt' % FLAGS.dataset))
            output_dir = FLAGS.predict_output_dir
            os.makedirs(output_dir, exist_ok=True)
            i = 0
            ds = dataset.get('predict')
            for batch in ds:
                predict = model.predict(batch)
                for p in predict:
                    if i % 1000 == 0:
                        logging.info('curr: %s/%s' % (i, len(files)))
                    p_r = tf.squeeze(tf.argmax(p, axis=-1)).numpy().astype('int16')
                    p_r = (p_r + 1) * 100
                    p_im = Image.fromarray(p_r)
                    im_path = os.path.join(output_dir, '%s.png' % files[i].split('/')[-1][:-4])
                    p_im.save(im_path)
                    i += 1
        if FLAGS.task == 'visualize_result':
            dataset.visualize_evaluate(model, FLAGS.mode)
    # delete_early_ckpt(model_dir)


# TODO： test的结果如何保存（HParams）


if __name__ == '__main__':
    app.run(main)
