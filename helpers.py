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
MODIFIED TO (NOT PRINT) AND (RETURN INDICATOR OF WINNER)
'''
def play_game_mod(board, p1, p2, q=False):
    """
    Runs Connect 4 game given simulator object and two agents (players)
    """
    reward = None

    if q == True:
        while True:
            #print("p1")
            p1move = p1.interact(reward, board)
            #print(p1move)
            if (p1move is None):
                board.print_grid()
                print("error player 1 a")
                return -1, 0
            p1result, reward = board.move(p1move, 1)
            #print p1result
            if (p1result == 1):
                #print("player 1")
                #board.print_grid()
                return 1, 1
            elif (p1result == -1):
                board.print_grid()
                print("error player 1 b")
                return -1, 0
            #print("p2")
            p2move = p2.calc_next_move()
            #print(p2move)
            if (p2move is None):
                board.print_grid()
                print("error player 2")
                return -1, 0
            p2result = board.move(p2move, -1)
            #print p2result
            if (p2result[0] == 1):
                #print("player 2")
                #board.print_grid()
                return 1, -1
            elif (p2result[0] == -1):
                board.print_grid()
                print("error player 2")
                return -1, 0

    else:
        while True:
            #print("p1")
            p1move = p1.calc_next_move()
            #print(p1move, board.next_possible_moves())
            #print(p1move)
            if (p1move is None):
                board.print_grid()
                print("error 1")
                return -1, 0
            p1result = board.move(p1move, 1)
            #print p1result
            if (p1result[0] == 1):
                #print("player 1")
                #board.print_grid()
                return 1, 1
            elif (p1result[0] == -1):
                board.print_grid()
                print("error 2")
                return -1, 0
            #print("p2")
            p2move = p2.calc_next_move()
            #print(p2move, board.next_possible_moves())
            #print(p2move)
            if (p2move is None):
                board.print_grid()
                print("error 3")
                return -1, 0
            p2result = board.move(p2move, -1)
            #print p2result
            if (p2result[0] == 1):
                #print("player 2")
                #board.print_grid()
                return 1, -1
            elif (p2result[0] == -1):
                board.print_grid()
                print("error 4")
                return -1, 0


