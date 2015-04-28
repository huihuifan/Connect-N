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
def play_game_no_output(board, p1, p2, q=False):
    """
    Runs Connect 4 game given simulator object and two agents (players)
    """
    reward = None
    
    while True:
        p1move = p1.calc_next_move(reward, board)
        if (p1move is None):
            return -1
        p1result, reward = board.move(p1move, 1)
        if (p1result == 1):
            p1.calc_next_move(reward, board)
            return 1
        elif (p1result == -1):
            return -1
        p2move = p2.calc_next_move(reward, board)
        if (p2move is None):
            return -1
        p2result = board.move(p2move, -1)
        if (p2result[0] == 1):
            return 2
        elif (p2result[0] == -1):
            return -1




