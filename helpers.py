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

from simulator import *
from agents import *

'''
Modified to not print and return the player who won
'''
def play_game_no_output(board, p1, p2):
    """
    Runs Connect 4 game given simulator object and two agents (players)
    """
    reward = None
    is_none = True
    last_board = None

    while True:
        last_board_1 = deepcopy(board)
        p1move = p1.calc_next_move(reward, board)
        if (p1move is None):
            return -1
        p1result, reward = board.move(p1move, 1)
        if (p1result == 1):
            p1.calc_next_move(reward, board)
            if (type(p2) is Q_Learner):
                p2.last_board_state = last_board_2.grid
                p2.last_action = p2move
                p2.calc_next_move(-1*reward, board)
            return 1
        elif (p1result == -1):
            return -1
        last_board_2 = deepcopy(board)
        if is_none:
            reward = None
            is_none = False
        p2move = p2.calc_next_move(reward, board)
        if (p2move is None):
            return -1
        p2result, reward = board.move(p2move, -1)
        if (p2result == 1):
            p2.calc_next_move(reward, board)
            if (type(p1) is Q_Learner):
                p1.last_board_state = last_board_1.grid
                p1.last_action = p1move
                p1.calc_next_move(-1*reward, board)
            return 2
        elif (p2result == -1):
            return -1



def run_many_games(x, p1, p2, games):
    """
    Run multiple games
    """
    p1_wins = 0
    p2_wins = 0
    draws = 0
    history = []
    for i in xrange(0,games):
        x.reset()
        winner = play_game_no_output(x, p1, p2)
        if winner == 1:
            p1_wins = p1_wins + 1
            history.append(1)
        if winner == 2:
            p2_wins = p2_wins + 1
            history.append(-1)
        if winner == -1:
            draws = draws + 1
            history.append(0)
        #print(winner)
    #print(p1_wins, p2_wins, draws)
    return p1_wins, p2_wins, draws, history


def parallel_MCTS_explore(i):
    """
    Helper for running the MCTS exploration term GridSearch in parallel
    """
    games = 100
    x = ConnectN(5, 3)
    p1 = Random_Learner(x)
    p2 = MCTS(x, 100,i)
    #p2 = Random_Learner(x)
    p1_wins, p2_wins, draws, history = run_multiple_games(x, p1, p2, games)
    #print(i)
    return p2_wins/games


def select_MCTS_exploreterm():
    """
    Runs the MCTS exploration term GridSearch
    """
    exp_term_range = np.array(range(0,11))/10.0
    if __name__ == '__main__':
        pool = Pool()  # start all workers
        win_rate = pool.map(parallel_MCTS_explore, exp_term_range)
        win_rate = np.array(win_rate)
    plt.plot(exp_term_range, win_rate)
    plt.title('MCTS Exploration Term Grid Search')
    plt.xlabel('Exploration Term Value')
    plt.ylabel('Win-rate against Random_Learner')
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.grid()
    plt.savefig('MCTS_param.png', dpi=1000)
    return exp_term_range, win_rate



