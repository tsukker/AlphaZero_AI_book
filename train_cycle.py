'''
1. start
2. create dual network
3. create training data by self-play
4. update parameters by created traing data
5. evaluate updated parameters
6. back to 3. (self-play) while count of learning cycle is less than 10
7. finish
'''

from dual_network import dual_network
from self_play import self_play
from train_network import train_network
from evaluate_network import evaluate_network
from evaluate_best_player import evaluate_best_player

if __name__ == '__main__':
    dual_network()
    for i in range(10):
        print('Train', i, '====================')
        self_play()
        train_network()
        update_best_player = evaluate_network()
        if update_best_player:
            evaluate_best_player()
