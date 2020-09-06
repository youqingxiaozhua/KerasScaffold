from tensorflow import keras
from tensorflow.keras import layers as l
from absl import flags
FLAGS = flags.FLAGS

__all__ = ['basic_rnn', ]


def basic_rnn(*args, **kwargs):
    sequence_input = l.Input(shape=(55, ), name='sequence')
    charge_input = l.Input(shape=(1,), name='charge')
    x = l.Embedding(20, 16)(sequence_input)
    x = l.Bidirectional(l.LSTM(64))(x)
    x = l.Dense(16, activation='relu')(x)
    x_cont = l.concatenate([x, charge_input])
    x = l.Dense(8, activation='relu')(x_cont)

    outputs = l.Dense(1, activation='sigmoid', name='predictions')(x)

    model = keras.Model(inputs=(sequence_input, charge_input), outputs=outputs)
    return model


# model = basic_rnn()
# keras.utils.plot_model(model, "multi_input_and_output_model.png", show_shapes=True)
