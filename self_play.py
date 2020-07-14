from datetime import datetime
import os
from pathlib import Path
import pickle

import numpy as np
import tensorflow as tf

from dual_network import DN_OUTPUT_SIZE
from game import State
from pv_mcts import pv_mcts_scores

SP_GAME_COUNT = 500  # number of games self-played, AlphaZero: 25000
SP_TEMPERATURE = 1.0  # temp. param of boltzman distribution


def first_player_value(ended_state):
    if ended_state.is_lose():
        return -1 if ended_state.is_first_player() else 1
    return 0


def dump_data(history):
    os.makedirs('./data/', exist_ok=True)
    filename = datetime.now().strftime('%Y%m%d%H%M%S') + '.history'
    path = './data/' + filename
    with open(path, mode='wb') as f:
        pickle.dump(history, f)


def play(model):
    history = []
    state = State()
    while True:
        if state.is_done():
            break
        scores = pv_mcts_scores(model, state, SP_TEMPERATURE)

        # Add state and policy into history
        policies = [0 for _ in range(DN_OUTPUT_SIZE)]
        for action, policy in zip(state.legal_actions(), scores):
            policies[action] = policy
        history.append([[state.pieces, state.enemy_pieces], policies, None])

        action = np.random.choice(state.legal_actions(), p=scores)
        state = state.next(action)
    value = first_player_value(state)
    for i in range(len(history)):
        history[i][2] = value
        value = -value
    return history


def self_play():
    history = []
    model = tf.keras.models.load_model('./model/best.h5')
    for i in range(SP_GAME_COUNT):
        h = play(model)
        history.extend(h)
        print('\rSelf-play {}/{}'.format(i + 1, SP_GAME_COUNT), end='')
    print('')
    dump_data(history)
    tf.keras.backend.clear_session()
    del model


if __name__ == '__main__':
    self_play()
