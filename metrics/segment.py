import tensorflow as tf
from sklearn.metrics import confusion_matrix
import numpy as np


class MeanIoU(tf.keras.metrics.MeanIoU):
    def update_state(self, y_true, y_pred, sample_weight=None):
        return super().update_state(tf.argmax(y_true, axis=-1), tf.argmax(y_pred, axis=-1), sample_weight)

    def cal_value(self, y_true, y_pred, one_hot=True):
        if one_hot:
            y_true = tf.argmax(y_true, axis=-1).numpy()
            y_pred = tf.argmax(y_pred, axis=-1).numpy()
        y_pred = y_pred.flatten()
        y_true = y_true.flatten()
        current = confusion_matrix(y_true, y_pred)
        # compute mean iou
        intersection = np.diag(current)
        ground_truth_set = current.sum(axis=1)
        predicted_set = current.sum(axis=0)
        union = ground_truth_set + predicted_set - intersection
        IoU = intersection / union.astype(np.float32)
        return np.mean(IoU)