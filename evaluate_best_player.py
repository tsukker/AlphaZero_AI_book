from pathlib import Path

import numpy as np
import tensorflow as tf

from game import State, random_action, alpha_beta_action, mcts_action
from pv_mcts import pv_mcts_action

EP_GAME_COUNT = 50


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


def evaluate_algorithm_of(label, next_actions):
    reversed_actions = list(reversed(next_actions))
    total_point = 0
    for i in range(EP_GAME_COUNT):
        if i % 2 == 0:
            total_point += play(next_actions)
        else:
            total_point += 1 - play(reversed_actions)
        print('\rEvaluate {}/{}'.format(i + 1, EP_GAME_COUNT), end='')
    print('')
    average_point = total_point / EP_GAME_COUNT
    print(label, average_point)


def evaluate_best_player():
    model = tf.keras.models.load_model('./model/best.h5')
    next_pv_mcts_action = pv_mcts_action(model, 0.0)

    next_actions = [next_pv_mcts_action, random_action]
    evaluate_algorithm_of('VS_Random', next_actions)

    next_actions = [next_pv_mcts_action, alpha_beta_action]
    evaluate_algorithm_of('VS_AlphaBeta', next_actions)

    next_actions = [next_pv_mcts_action, mcts_action]
    evaluate_algorithm_of('VS_MCTS', next_actions)

    tf.keras.backend.clear_session()
    del model


if __name__ == '__main__':
    evaluate_best_player()
