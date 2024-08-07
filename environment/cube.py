import random
import copy
import numpy as np


class Cube():

    def __init__(self):
        self.state = np.arange(27).reshape(3, 3, 3)
        self.moves = [self.front, self.front_p, self.right, self.right_p,
                          self.up, self.up_p, self.left, self.left_p,
                          self.back, self.back_p, self.down, self.down_p]

    def __eq__(self, other):
        equal = False
        if isinstance(other, self.__class__):
            equal = (self.state == other.state).all()
        return equal

    def __ne__(self, other):
        return not self.__eq__(other)

    def front(self):
        # F
        self.state[0] = np.rot90(self.state[0], k=-1)
        return copy.deepcopy(self.state)

    def front_p(self):
        # F'
        self.state[0] = np.rot90(self.state[0], k=1)
        return copy.deepcopy(self.state)

    def right(self):
        # R
        self.state[:, :, 2] = np.rot90(self.state[:, :, 2], k=1)
        return copy.deepcopy(self.state)

    def right_p(self):
        # R'
        self.state[:, :, 2] = np.rot90(self.state[:, :, 2], k=-1)
        return copy.deepcopy(self.state)

    def up(self):
        # U
        self.state[:, 0, :] = np.rot90(self.state[:, 0, :], k=1)
        return copy.deepcopy(self.state)

    def up_p(self):
        # U'
        self.state[:, 0, :] = np.rot90(self.state[:, 0, :], k=-1)
        return copy.deepcopy(self.state)

    def left(self):
        # L
        self.state[:, :, 0] = np.rot90(self.state[:, :, 0], k=-1)
        return copy.deepcopy(self.state)

    def left_p(self):
        # L'
        self.state[:, :, 0] = np.rot90(self.state[:, :, 0], k=1)
        return copy.deepcopy(self.state)

    def back(self):
        # B
        self.state[2] = np.rot90(self.state[2], k=1)
        return copy.deepcopy(self.state)

    def back_p(self):
        # B'
        self.state[2] = np.rot90(self.state[2], k=-1)
        return copy.deepcopy(self.state)

    def down(self):
        # D
        self.state[:, 2, :] = np.rot90(self.state[:, 2, :], k=-1)
        return copy.deepcopy(self.state)

    def down_p(self):
        # D'
        self.state[:, 2, :] = np.rot90(self.state[:, 2, :], k=1)
        return copy.deepcopy(self.state)

    def shuffle(self, n, log=False):
        for _ in range(n):
            move = random.choice(self.moves)
            move()
            if log:
                print(move.__name__, end=' ')
        if log: print('')
        return copy.deepcopy(self.state)

    def set_state(self, new_state):
        self.state = copy.deepcopy(new_state)
    
    def get_state(self):
        return copy.deepcopy(self.state)
    
    def reset(self):
        self.state = np.arange(0, 27).reshape(3, 3, 3)
        return copy.deepcopy(self.state)