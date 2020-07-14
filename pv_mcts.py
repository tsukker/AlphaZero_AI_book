import math
from pathlib import Path

import numpy as np
import tensorflow as tf

from dual_network import DN_INPUT_SHAPE
from game import State

PV_EVALUATE_COUNT = 50  # AlphaZero: 1600


def predict(model: tf.keras.models.Model, state: State):
    x = np.array([state.pieces, state.enemy_pieces])
    x = x.reshape([DN_INPUT_SHAPE[-1]] + DN_INPUT_SHAPE[:-1]).transpose(
        1, 2, 0).reshape([1] + DN_INPUT_SHAPE)

    y = model.predict(x, batch_size=1)

    policies = y[0][0][list(state.legal_actions())]
    policies /= sum(policies) if sum(policies) > 0 else 1

    value = y[1][0][0]
    return policies, value


def nodes_to_scores(nodes):
    return [c.n for c in nodes]


def boltzman(xs, temperature):
    tmp = [x**(1 / temperature) for x in xs]
    sum_tmp = sum(tmp)
    return [x / sum_tmp for x in tmp]


def pv_mcts_scores(model, state, temperature):
    root_node = Node(state, 0)
    for _ in range(PV_EVALUATE_COUNT):
        root_node.evaluate(model)

    # probability distribution of legal actions
    scores = nodes_to_scores(root_node.child_nodes)
    if temperature == 0:
        action = np.argmax(scores)
        scores = np.zeros(len(scores))
        scores[action] = 1
    else:
        scores = boltzman(scores, temperature)
    return scores


class Node:
    def __init__(self, state, policy):
        self.state = state
        self.policy = policy
        self.w = 0
        self.n = 0
        self.child_nodes = None

    def evaluate(self, model):
        if self.state.is_done():
            value = -1 if self.state.is_lose() else 0
            self.w += value
            self.n += 1
            return value
        if not self.child_nodes:
            policies, value = predict(model, self.state)
            self.w += value
            self.n += 1

            self.child_nodes = [
                Node(self.state.next_state(action), policy)
                for action, policy in zip(self.state.legal_actions(), policies)
            ]
            return value
        else:
            value = -self.next_child_node().evaluate(model)
            self.w += value
            self.n += 1
            return value

    def next_child_node(self):
        C_PUCT = 1.0
        t = sum(nodes_to_scores(self.child_nodes))
        pucb_values = [
            (-child_node.w / child_node.n if child_node.n > 0 else 0.0) +
            C_PUCT * child_node.policy * math.sqrt(t) / (1 + child_node.n)
            for child_node in self.child_nodes
        ]
        return self.child_nodes[np.argmax(pucb_values)]


def pv_mcts_action(model, temperature=0):
    def pv_mcts_action(state):
        scores = pv_mcts_scores(model, state, temperature)
        return np.random.choice(state.legal_actions(), p=scores)

    return pv_mcts_action


if __name__ == '__main__':
    path = sorted(Path('./model').glob('*.h5'))[-1]
    model = tf.keras.models.load_model(str(path))
    state = State()
    next_action = pv_mcts_action(model, 1.0)

    while True:
        if state.is_done():
            break
        action = next_action(state)
        state = state.next_state(action)
        print(state)
