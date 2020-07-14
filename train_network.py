from pathlib import Path
import pickle

import numpy as np
import tensorflow as tf

from dual_network import DN_INPUT_SHAPE

RN_EPOCHS = 100  # count of learnings


def load_history():
    history_path = sorted(Path('./data').glob('*.history'))[-1]
    with history_path.open(mode='rb') as f:
        return pickle.load(f)


def train_network():
    history = load_history()
    xs, y_policies, y_values = zip(*history)
    xs = np.array(xs)
    a, b, c = DN_INPUT_SHAPE
    xs = xs.reshape([len(xs), c, a, b])
    xs = xs.transpose([0, 2, 3, 1])
    y_policies = np.array(y_policies)
    y_values = np.array(y_values)

    model = tf.keras.models.load_model('./model/best.h5')
    model.compile(loss=['categorical_crossentropy', 'mse'], optimizer='adam')

    def step_decay(epoch):
        x = 0.001
        if 50 <= epoch:
            x = 0.0005
        elif 80 <= epoch:
            x = 0.00025
        return x

    lr_decay = tf.keras.callbacks.LearningRateScheduler(step_decay)

    print_callback = tf.keras.callbacks.LambdaCallback(
        on_epoch_begin=lambda epoch, logs: print(
            '\rTrain {}/{}'.format(epoch + 1, RN_EPOCHS), end=''))

    model.fit(
        xs,
        [y_policies, y_values],
        batch_size=128,
        epochs=RN_EPOCHS,
        verbose=0,
        callbacks=[lr_decay, print_callback],
    )
    print('')
    model.save('./model/latest.h5')

    tf.keras.backend.clear_session()
    del model


if __name__ == '__main__':
    train_network()
