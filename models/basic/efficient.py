import efficientnet.tfkeras as efn
from tensorflow.keras import layers, Model


def EfficientNetB7(include_top=False,
                   weights='imagenet',
                   input_tensor=None,
                   input_shape=None,
                   pooling=None,
                   classes=1000, *args, **kwargs):
    model = efn.EfficientNetB7(
        include_top=False,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        classes=classes)  # or weights='noisy-student'
    # 一共806层
    if weights:
        for i in model.layers[:688]:
            i.trainable = False
    x = model.output
    if pooling == 'avg':
        x = layers.GlobalAveragePooling2D()(x)
    elif pooling == 'max':
        x = layers.GlobalMaxPooling2D()(x)
    else:
        x = layers.Flatten(name='flatten')(x)
        # x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
        activation = 'sigmoid' if classes == 1 else 'softmax'
        x = layers.Dense(classes, activation=activation, name='predictions')(x)
    model = Model(model.input, x, name='EfficientNetB7')

    return model
