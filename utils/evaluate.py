import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from metrics import MeanIoU


def evaluate_batch(model:tf.keras.Model, ds, images, masks=None, classes=2, plot_num=0, skip=0):
    """
    挨个预测
    :param model:
    :param images: image file list
    :param masks: mask file list
    :return: [(image, mask, predict, miou), ]
    """
    ds.color_map_viz()
    result = []
    stacked = [ds.one_hot(*ds.load(i,j)) for i,j in zip(images, masks)]
    for image, mask in stacked:
        predict = model.predict(tf.expand_dims(image, 0))
        m = MeanIoU(num_classes=classes)
        gdt = mask.numpy().reshape(1, *mask.shape)
        miou = m.cal_value(gdt, predict)
        if miou < 1.:
            result.append((image, tf.squeeze(tf.argmax(mask, axis=-1)), tf.squeeze(tf.argmax(predict, axis=-1)), miou))
    result.sort(key=lambda i: i[3])
    for i in result:
        print(i[3])
    if plot_num:
        def plot_cases(start_from_left=True):
            plt.figure()
            for index in range(plot_num):
                i = index + skip if start_from_left else -(index+1)-skip
                xlabel = {
                    0: 'image',
                    1: 'groundtruth',
                    2: 'mIou: %s' % result[i][3]
                }
                for j in range(3):
                    plt.subplot(plot_num, 3, index * 3 + j+1)
                    plt.imshow(result[i][j])
                    plt.xticks([])
                    plt.xlabel(xlabel[j])
                    plt.yticks([])
                    if j > 0:
                        plt.ylabel(np.unique(result[i][j]))
            plt.show()
        plot_cases(True)
        plot_cases(False)

    return result


