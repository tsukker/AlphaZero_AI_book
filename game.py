import math
import random


class State:
    def __init__(self, pieces=None, enemy_pieces=None):
        self.pieces = pieces if pieces != None else [0 for _ in range(9)]
        self.enemy_pieces = enemy_pieces if enemy_pieces != None else [
            0 for _ in range(9)
        ]

    def piece_count(self, pieces):
        return sum(pieces)

    def is_lose(self):
        def is_comp(x, y, dx, dy):
            for _ in range(3):
                if y < 0 or 3 <= y or x < 0 or 3 <= x or self.enemy_pieces[
                        y * 3 + x] == 0:
                    return False
                x, y = x + dx, y + dy
            return True

        if is_comp(0, 0, 1, 1) or is_comp(0, 2, 1, -1):
            return True
        for i in range(3):
            if is_comp(0, i, 1, 0) or is_comp(i, 0, 0, 1):
                return True
        return False

    def is_draw(self):
        return (not self.is_lose()) and sum(self.pieces) + sum(
            self.enemy_pieces) == 9

    def is_done(self):
        return self.is_lose() or self.is_draw()

    def next_state(self, action):
        pieces = self.pieces.copy()
        pieces[action] = 1
        return State(self.enemy_pieces, pieces)

    def legal_actions(self):
        actions = []
        for i in range(9):
            if self.pieces[i] == 0 and self.enemy_pieces[i] == 0:
                actions.append(i)
        return actions

    def is_first_player(self):
        return sum(self.pieces) == sum(self.enemy_pieces)

    def __str__(self):
        ox = ('o', 'x') if self.is_first_player() else ('x', 'o')
        s = ''
        for i in range(9):
            if self.pieces[i] == 1:
                s += ox[0]
            elif self.enemy_pieces[i] == 1:
                s += ox[1]
            else:
                s += '-'
            if i % 3 == 2:
                s += '\n'
        return s


def random_action(state):
    legal_actions = state.legal_actions()
    return legal_actions[random.randint(0, len(legal_actions) - 1)]


def alpha_beta(state, alpha, beta):
    if state.is_lose():
        return -1
    if state.is_draw():
        return 0
    for action in state.legal_actions():
        score = -alpha_beta(state.next_state(action), -beta, -alpha)
        if alpha < score:
            alpha = score
        if beta <= alpha:
            return alpha
    return alpha


def alpha_beta_action(state):
    best_action = 0
    alpha = -float('inf')
    s = ['', '']
    for action in state.legal_actions():
        score = -alpha_beta(state.next_state(action), -float('inf'), -alpha)
        if alpha < score:
            best_action = action
            alpha = score
        s[0] = '{}{:2d}, '.format(s[0], action)
        s[1] = '{}{:2d}, '.format(s[1], score)
    #print('action: ', s[0])
    #print('score: ', s[1])
    return best_action


def playout(state):
    if state.is_lose():
        return -1
    if state.is_draw():
        return 0
    return -playout(state.next_state(random_action(state)))


def argmax(collection, key=None):
    return collection.index(max(collection))


def mcs_action(state):
    legal_actions = state.legal_actions()
    values = [0 for _ in range(len(legal_actions))]
    for i, action in enumerate(legal_actions):
        for _ in range(10):
            values[i] += -playout(state.next_state(action))
    return legal_actions[argmax(values)]


class Node:
    def __init__(self, state):
        self.state = state
        self.w = 0
        self.n = 0
        self.child_nodes = None

    def evaluate(self):
        if self.state.is_done():
            value = -1 if self.state.is_lose() else 0
            self.w += value
            self.n += 1
            return value
        if self.child_nodes is None:
            value = playout(self.state)
            self.w += value
            self.n += 1
            if self.n == 10:
                self.expand()
            return value
        else:
            value = -self.next_child_node().evaluate()
            self.w += value
            self.n += 1
            return value

    def expand(self):
        self.child_nodes = [
            Node(self.state.next_state(action))
            for action in self.state.legal_actions()
        ]

    def next_child_node(self):
        t = 0
        for child_node in self.child_nodes:
            if child_node.n == 0:
                return child_node
            t += child_node.n
        ucb1 = [
            -node.w / node.n + (2 * math.log(t) / node.n)**0.5
            for node in self.child_nodes
        ]
        return self.child_nodes[argmax(ucb1)]


def mcts_action(state):
    root_node = Node(state)
    root_node.expand()
    for _ in range(100):
        root_node.evaluate()
    legal_actions = state.legal_actions()
    n_list = [c.n for c in root_node.child_nodes]
    return legal_actions[argmax(n_list)]


if __name__ == '__main__':
    state = State()
    while True:
        if state.is_done():
            break
        state = state.next_state(random_action(state))
        print(state)
