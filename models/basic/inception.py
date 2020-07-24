from tensorflow.keras.applications import inception_v3

__all__ = ('InceptionV3', )

from tensorflow.keras import Model, layers


def InceptionV3(include_top=False,
                weights='imagenet',
                input_tensor=None,
                input_shape=None,
                pooling=None,
                classes=1000, *args, **kwargs):
    model = inception_v3.InceptionV3(
        include_top=False,
        weights=weights,
        input_shape=input_shape,
        input_tensor=input_tensor,
        pooling=pooling,
        classes=classes
    )
    # 一共311层
    if weights:
        for i in model.layers[:288]:
            i.trainable = False
    x = model.output
    if pooling == 'avg':
        x = layers.GlobalAveragePooling2D()(x)
    elif pooling == 'max':
        x = layers.GlobalMaxPooling2D()(x)
    else:
        x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
        activation = 'sigmoid' if classes == 1 else 'softmax'
        x = layers.Dense(classes, activation=activation, name='predictions')(x)
    model = Model(model.input, x, name='InceptionV3')
    return model
