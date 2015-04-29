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


class ConnectN:
    """
    Connect N game simulator for two players, 1 and -1.

    Inputs:
    Grid size- creates a grid size x grid size square board
    N- number of tokens a player must connect to win the game
    """

    def __init__(self, grid_size, n):
        self.n = n
        self.grid_size = grid_size

        # initializes board with zero, which represents empty token
        self.grid = np.zeros([grid_size,grid_size])
        self.finished = 0
        self.turn_num = 0

    def reset(self):
        self.__init__(self.grid_size, self.n)

    def check_win(self, col, row, player):
        """
        Checks if given player has connected N tokens.
        """
        for i in range(0, self.n):
            if sum(self.grid[col, row - i:row - i + self.n]) == self.n*player:
                self.finished = 1
                return 1
            if sum(self.grid[col - i: col - i + self.n, row]) == self.n*player:
                self.finished = 1
                return 1
            if col - i >= 0 and col - i + self.n - 1 < self.grid_size and row - i >= 0 and row - i + self.n - 1 < self.grid_size:
                if sum([self.grid[col - i + x, row - i + x] for x in range(0, self.n)]) == self.n*player:
                    self.finished = 1
                    return 1
            if col - i >= 0 and col - i + self.n - 1 < self.grid_size and row + i >= self.n - 1 and row + i < self.grid_size:
                if sum([self.grid[col - i + x, row + i - x] for x in range(0, self.n)]) == self.n*player:
                    self.finished = 1
                    return 1
        return 0

    def move(self, col, player):
        """
        Given player and column to move in, modifies board and increments the turn counter.

        Returns a tuple, where first value is return message and second value is reward.
        """
        self.turn_num += 1

        if self.finished == 1:
            return 1, 50
        sum_col = np.sum([abs(x) for x in self.grid[col]])
        if sum_col == self.grid_size:
            return -1, -1
        self.grid[col, sum_col] = player
        if self.check_win(col, sum_col, player) == 1:
            return 1, 50
        return 0, 0

    def simulate_move(self, col, player):
        """
        Tests a move and returns if it is valid or not
        """
        sum_col = np.sum([abs(x) for x in self.grid[col]])
        if sum_col == self.grid_size:
            return 1
        else:
            return 0

    def turn(self):
        """
        Returns which player's turn it is. First turn is player 1, second turn is player -1.
        """
        if self.turn_num%2 == 0:
            return 1
        else:
            return -1

    def next_possible_moves(self):
        """
        Returns array of possible columns for a next move
        """
        columns = []

        for i in xrange(0, self.grid_size):
            if (0 in self.grid[i]):
                columns.append(i)

        return columns

    def all_tokens_placed(self):
        """
        Returns location of all tokens (column, row) that have been placed
        """
        all_tokens = []

        for col in xrange(0, self.grid_size):
            for row in xrange(0, self.grid_size):
                if self.grid[col][row] != 0:
                    all_tokens.append({"location": [col, row], "player": self.grid[col][row]})

        return all_tokens

    def is_empty(self, col, row):
        """
        Returns if a given spot (column, row) is empty
        """
        return self.grid[col][row] == 0

    """
    Following streak functions check if player has token streak in the four possible win directions
    """
    def streakVertical(self, board, col, row, player):
        if row < 0 or row > len(board[col]) - self.n or col < 0 or col >= self.grid_size:
            return 0
        for i in range(0,self.n):
            if board[col][row + i] == -1*player:
                return 0
            if board[col][row + i] == 0:
                return i
        return self.n

    def streakHorizontal(self, board, col, row, player):
        if row < 0 or row >= self.grid_size or col < 0 or col > len(board) - self.n:
            return 0
        for i in range(0,self.n):
            if board[col + i][row] == -1*player:
                return 0
            if board[col + i][row] == 0:
                return i
        return self.n

    def streakDiagonalUp(self, board, col, row, player):
        if row < 0 or row > len(board[col]) - self.n or col < 0 or col > len(board) - self.n:
            return 0
        for i in range(0,self.n):
            if board[col + i][row + i] == -1*player:
                return 0
            if board[col + i][row + i] == 0:
                return i
        return self.n

    def streakDiagonalDown(self, board, col, row, player):
        if row < self.n or row >= self.grid_size or col > len(board) - self.n or col < 0:
            return 0
        for i in range(0,self.n):
            if board[col + i][row - i] == -1*player:
                return 0
            if board[col + i][row - i] == 0:
                return i
        return self.n

    def print_grid(self):
        print(np.rot90(self.grid))


