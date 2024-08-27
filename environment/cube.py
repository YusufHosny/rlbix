import random
import copy
import numpy as np


class Cube():

    def __init__(self):
        self.state = np.zeros((6, 5, 5, 5), dtype='float32') # WYGBOR color order
        self.state[0, 1:4, 1:4, 4] += 1. # top white
        self.state[1, 1:4, 1:4, 0] += 1. # bottom yellow
        self.state[2, 1:4, 0, 1:4] += 1. # front green
        self.state[3, 1:4, 4, 1:4] += 1. # back blue
        self.state[4, 0, 1:4, 1:4] += 1. # left orange
        self.state[5, 4, 1:4, 1:4] += 1. # right red
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
        for channel in self.state:
            channel[:, 1] = np.rot90(channel[:, 1], k=-1)
        return copy.deepcopy(self.state)

    def front_p(self):
        # F'
        for channel in self.state:
            channel[:, 1] = np.rot90(channel[:, 1], k=1)
        return copy.deepcopy(self.state)

    def right(self):
        # R
        for channel in self.state:
            channel[3] = np.rot90(channel[3], k=-1)
        return copy.deepcopy(self.state)

    def right_p(self):
        # R'
        for channel in self.state:
            channel[3] = np.rot90(channel[3], k=1)
        return copy.deepcopy(self.state)

    def up(self):
        # U
        for channel in self.state:
            channel[:, :, 3] = np.rot90(channel[:, :, 3], k=-1)
        return copy.deepcopy(self.state)

    def up_p(self):
        # U'
        for channel in self.state:
            channel[:, :, 3] = np.rot90(channel[:, :, 3], k=1)
        return copy.deepcopy(self.state)

    def left(self):
        # L
        for channel in self.state:
            channel[1] = np.rot90(channel[1], k=1)
        return copy.deepcopy(self.state)

    def left_p(self):
        # L'
        for channel in self.state:
            channel[1] = np.rot90(channel[1], k=-1)
        return copy.deepcopy(self.state)

    def back(self):
        # B
        for channel in self.state:
            channel[:, 3] = np.rot90(channel[:, 3], k=1)
        return copy.deepcopy(self.state)

    def back_p(self):
        # B'
        for channel in self.state:
            channel[:, 3] = np.rot90(channel[:, 3], k=-1)
        return copy.deepcopy(self.state)

    def down(self):
        # D
        for channel in self.state:
            channel[:, :, 1] = np.rot90(channel[:, :, 1], k=1)
        return copy.deepcopy(self.state)

    def down_p(self):
        # D'
        for channel in self.state:
            channel[:, :, 1] = np.rot90(channel[:, :, 1], k=-1)
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
        self.state = np.zeros((6, 5, 5, 5), dtype='float32') # WYGBOR color order
        self.state[0, 1:4, 1:4, 4] += 1. # top white
        self.state[1, 1:4, 1:4, 0] += 1. # bottom yellow
        self.state[2, 1:4, 0, 1:4] += 1. # front green
        self.state[3, 1:4, 4, 1:4] += 1. # back blue
        self.state[4, 0, 1:4, 1:4] += 1. # left orange
        self.state[5, 4, 1:4, 1:4] += 1. # right red
        return copy.deepcopy(self.state)