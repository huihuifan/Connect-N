# imports

from __future__ import division
import numpy as np
from numpy.random import rand
import math
from random import randint
import itertools
import random
import copy
from copy import deepcopy
from multiprocessing import Pool
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')

import simulator
import agents

'''
Modified to not print and return the player who won
'''
def play_game_mod(board, p1, p2, q=False):
    """
    Runs Connect 4 game given simulator object and two agents (players)

    Returns player number who has won
    """
    reward = None

    while True:
        p1move = p1.interact(reward, board)
        if (p1move is None):
            board.print_grid()
            print("error player 1 a")
            return -1, 0
        p1result, reward = board.move(p1move, 1)
        if (p1result == 1):
            return 1, 1
        elif (p1result == -1):
            board.print_grid()
            print("error player 1 b")
            return -1, 0
        p2move = p2.calc_next_move()
        if (p2move is None):
            board.print_grid()
            print("error player 2")
            return -1, 0
        p2result = board.move(p2move, -1)
        if (p2result[0] == 1):
            return 1, -1
        elif (p2result[0] == -1):
            board.print_grid()
            print("error player 2")
            return -1, 0



