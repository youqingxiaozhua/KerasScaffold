import tensorflow as tf

from tensorflow.python.keras.callbacks import ModelCheckpoint, TensorBoard, LambdaCallback, EarlyStopping


def model_checkpoint(filepath, monitor='val_loss'):
    return ModelCheckpoint(
        filepath=filepath,
        monitor=monitor,
        verbose=1,
        save_best_only=True,
        save_weights_only=True,
        # save_freq=save_freq, , save_freq=dataset.train_size * FLAGS.save_freq
    )


def tensorboard(log_dir):
    return TensorBoard(
        log_dir=log_dir,
        # histogram_freq=5,
        # write_images=True,
    )


def save_predict_image(test_img, exp_dir, model):
    def save_predict(image, logdir, model, file_writer_image):
        def callback(epoch, logs):
            im = tf.expand_dims(image, axis=0)
            if epoch <= 1:
                with file_writer_image.as_default():
                    tf.summary.image("Origin", im, step=epoch)
            # tf.print('before predict', tf.shape(im))
            predict = model.predict(im)
            # tf.print('after predict', tf.shape(predict))
            # predict = tf.argmax(predict, axis=-1)
            predict = VOCColormap(21).colorize(predict)
            with file_writer_image.as_default():
                tf.summary.image("Predict", predict, step=epoch)
        return callback

    file_writer_image = tf.summary.create_file_writer(exp_dir + '/image')
    return LambdaCallback(
        on_epoch_end=save_predict(test_img, exp_dir, model, file_writer_image))


def early_stopping(monitor='val_accuracy',  patience=20):
    return EarlyStopping(
        monitor=monitor, min_delta=0.0005, patience=patience, verbose=1, mode='auto',
        restore_best_weights=True,
    )


def lr_schedule(name, epochs=200):
    def poly(epoch, lr):
        return lr * (1- epoch/epochs) ** 0.9

    def constant(epoch, lr):
        return lr
    schedulers = {
        'poly': poly,
        'constant': constant
    }
    return schedulers[name]



