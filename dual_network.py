import os

import tensorflow as tf

DN_FILTERS = 128
DN_RESIDUAL_NUM = 16
DN_INPUT_SHAPE = (3, 3, 2)
DN_OUTPUT_SIZE = 9


def conv(filters):
    return tf.keras.layers.Conv2D(
        filters,
        3,
        padding='same',
        use_bias=False,
        kernel_initializer='he_normal',
        kernel_regularizer=tf.keras.regularizers.l2(0.0005),
    )


def residual_block():
    def f(x):
        short_cut = x
        x = conv(DN_FILTERS)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = conv(DN_FILTERS)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Add()([x, short_cut])
        x = tf.keras.layers.Activation('relu')(x)
        return x

    return f


def dual_network():
    if os.path.exists('./model/best.h5'):
        return

    # input
    input_layer = tf.keras.layers.Input(shape=DN_INPUT_SHAPE)

    # convolution layer
    x = conv(DN_FILTERS)(input_layer)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    # residual block * DN_RESIDUAL_NUM
    for _ in range(DN_RESIDUAL_NUM):
        x = residual_block()(x)

    # pooling layer
    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    # policy output
    p = tf.keras.layers.Dense(
        DN_OUTPUT_SIZE,
        activation='softmax',
        kernel_regularizer=tf.keras.regularizers.l2(0.0005),
        name='pi',
    )(x)

    # value output
    v = tf.keras.layers.Dense(
        1,
        kernel_regularizer=tf.keras.regularizers.l2(0.0005),
    )(x)
    v = tf.keras.layers.Activation('tanh', name='v')(v)

    model = tf.keras.models.Model(inputs=input_layer, outputs=[p, v])
    os.makedirs('./model/', exist_ok=True)
    model.save('./model/best.h5')

    tf.keras.backend.clear_session()
    del model


if __name__ == '__main__':
    dual_network()
