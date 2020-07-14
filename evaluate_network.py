from pathlib import Path
import shutil

import numpy as np
import tensorflow as tf

from game import State
from pv_mcts import pv_mcts_action

EN_GAME_COUNT = 100  # count of games per evaluation, AlphaZero: 400
EN_TEMPERATURE = 1.0  # temp. param of boltzman distribution


def first_player_point(ended_state):
    if ended_state.is_lose():
        return 0 if ended_state.is_first_player() else 1
    return 0.5


def play(next_actions):
    state = State()
    while True:
        if state.is_done():
            break
        action_idx = 0 if state.is_first_player() else 1
        next_action = next_actions[action_idx]
        action = next_action(state)
        state = state.next_state(action)
    return first_player_point(state)


def update_best_player():
    shutil.copy('./model/latest.h5', './model/best.h5')
    print('Update BestPlayer')


def evaluate_network():
    model_paths = ['./model/latest.h5', './model/best.h5']
    models = [tf.keras.models.load_model(path) for path in model_paths]
    next_actions = [pv_mcts_action(model, EN_TEMPERATURE) for model in models]
    reversed_actions = list(reversed(next_actions))

    total_point = 0
    for i in range(EN_GAME_COUNT):
        if i % 2 == 0:
            total_point += play(next_actions)
        else:
            total_point += 1 - play(reversed_actions)
        print('\rEvaluate {}/{}'.format(i + 1, EN_GAME_COUNT), end='')
    print('')
    average_point = total_point / EN_GAME_COUNT
    print('Average point: ', average_point)

    tf.keras.backend.clear_session()
    del models

    if 0.55 < average_point:
        update_best_player()
        return True
    else:
        return False


if __name__ == '__main__':
    evaluate_network()
