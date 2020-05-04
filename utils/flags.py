from absl import flags


def define_flages():
    flags.DEFINE_integer('batch_size', 8, 'batch size')
    flags.DEFINE_integer('epoch', 100, 'epoch number')
    flags.DEFINE_integer('classes', None, 'class number')
    flags.DEFINE_float('lr', 0.001, 'learning rate')
    flags.DEFINE_bool('resume', True, 'if resume to continue')
    flags.DEFINE_string('exp_name', None, 'exp name')
    flags.DEFINE_string('dataset', None, 'dataset name, relative to ./dataset')
    flags.DEFINE_string('model', None, 'model name')
    flags.DEFINE_enum('mode', 'train', ('train', 'train_test'), 'run mode')

    flags.DEFINE_multi_integer('input_shape', None, 'model input shape')
    flags.DEFINE_integer('early_stopping_patience', 100, 'early_stopping_patience')
    flags.DEFINE_enum('model_checkpoint_monitor', 'val_accuracy', ('val_accuracy', 'val_loss'),
                      'early_stopping_patience')
    flags.DEFINE_bool('debug', False, 'debug mode verbose will set to 1(interact mode)')
    # model
    flags.DEFINE_string('weights', None, 'pretrain weights')
    return flags
