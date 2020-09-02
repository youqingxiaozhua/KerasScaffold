import os

import tensorflow as tf
import numpy as np
from absl import flags, logging
FLAGES = flags.FLAGS

AUTOTUNE = tf.data.experimental.AUTOTUNE
BASE_DIR = os.path.dirname(__file__)

SequenceMaxLength = 55
AminoAcidType = 20
AminoAcidList = 'ARNDCEQGHILKMFPSTWYV'
AminoAcidIndex = dict()
for i in range(len(AminoAcidList)):
    AminoAcidIndex[AminoAcidList[i]] = i


class DataLoader:
    def __init__(self, *args, **kwargs):
        self.train_size, self.valid_size, self.test_size = 0, 0, 0

    # pre process for one item
    def pre_process(self, data_pair):
        # data_pair = data_pair.numpy()
        # for i in range(3):
        #     data_pair[i] = str(data_pair[i], 'utf-8')
        sequence = data_pair[:, 0]
        charge = data_pair[:, 1]
        v = data_pair[:, 2]
        seq_index = list()
        for j in sequence:
            # seq_index.append(tf.one_hot([AminoAcidIndex[i] for i in j], AminoAcidType))
            seq_index.append([AminoAcidIndex[i] for i in j])
        # sequence = tf.one_hot(seq_index, 20)
        padded_sequence = tf.keras.preprocessing.sequence.pad_sequences(
            seq_index, maxlen=SequenceMaxLength, padding="post")

        v = v.astype(np.float)
        charge = charge.astype(np.float)
        # charge = int(charge)
        # assert 20 <= v <= 70

        v = v / 100.
        charge = charge / 8.

        return padded_sequence, charge, v

    def get(self, mode):
        data = np.loadtxt(os.path.join(BASE_DIR, 'data_processed.csv'), str, delimiter=',')
        train_size = 50000

        # data = data[-1000:]
        # train_size = 5

        sequence, charge, v = self.pre_process(data)


        data = {
            'train': (sequence[:train_size], charge[:train_size], v[:train_size]),
            'valid': (sequence[train_size:], charge[train_size:], v[train_size:]),
            'test': ([], [], [])
        }
        x1, x2, y = data[mode]

        setattr(self, '%s_size'%mode, len(x1))
        logging.info('%s_x1 shape: %s' % (mode, tf.shape(x1)))
        ds = tf.data.Dataset.from_tensor_slices(((x1, x2), y))
        ds = ds.shuffle(len(x1))
        # if mode == 'train':
        #     ds.repeat()
        batch_size = FLAGES.batch_size if 'batch_size' in FLAGES.flag_values_dict() else 8
        ds = ds.batch(batch_size)

        ds = ds.prefetch(AUTOTUNE)
        return ds


def pre_process():
    data = np.loadtxt(os.path.join(BASE_DIR, 'data.csv'), str, delimiter=',', skiprows=1)
    output = dict()
    for row in data:
        sequence, charge, cv, intensity = row
        item_key = (sequence, charge)
        if item_key not in output:
            output[item_key] = (cv, intensity)
        else:
            old_cv, old_intensity = output[item_key]
            if old_intensity < intensity:
                output[item_key] = (cv, intensity)
    with open(os.path.join(BASE_DIR, 'data_processed.csv'), 'w') as f:
        for key, value in output.items():
            f.write('%s,%s,%s\n' % (key[0], key[1], value[0]))


if __name__ == '__main__':
    # pre_process()
    from utils import flags
    flags.define_flages()

    ds = DataLoader()
    ds.get('train')
