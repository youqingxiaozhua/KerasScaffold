from absl import flags


def define_flages():
    flags.DEFINE_integer('batch_size', 8, 'batch size')
    flags.DEFINE_integer('epoch', 100, 'epoch number')
    flags.DEFINE_integer('classes', None, 'class number')
    flags.DEFINE_float('lr', 0.001, 'learning rate')
    flags.DEFINE_float('weight_decay', 0.00004, 'weight_decay')
    flags.DEFINE_string('resume', 'ckpt', '"ckpt" for resume from checkpoint, or file path to weights file')
    flags.DEFINE_string('exp_name', None, 'exp name')
    flags.DEFINE_string('dataset', None, 'dataset name, relative to ./dataset')
    flags.DEFINE_string('model', None, 'model name')  # must be define in /models/__init__.py
    flags.DEFINE_string('loss', 'categorical_crossentropy', 'loss name')  # must be define in /losses/__init__.py
    flags.DEFINE_string('mode', 'train', 'dataset file list')
    flags.DEFINE_enum('task', None, ('visualize_result', ), 'task expect train')

    flags.DEFINE_multi_integer('input_shape', None, 'model input shape')
    flags.DEFINE_integer('early_stopping_patience', 100, 'early_stopping_patience')
    flags.DEFINE_enum('model_checkpoint_monitor', 'val_accuracy', ('val_accuracy', 'val_loss'),
                      'early_stopping_patience')
    flags.DEFINE_string('lr_schedule', 'constant', 'lr_schedule')
    flags.DEFINE_bool('debug', False, 'debug mode verbose will set to 1(interact mode)')
    # model
    flags.DEFINE_string('weights', None, 'pretrain weights')
    flags.DEFINE_integer('freeze_layers', 0, 'freeze first n layers')
    flags.DEFINE_string('predict_output_dir', None, 'predict output dir')
    return flags
