import numpy as np
from numpy.random import rand
import math
from random import randint
import itertools
import random
import copy

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

    def print_grid(self):
        print(np.rot90(self.grid))


class Minimax_Learner(object):
    """
    Implementation of AI algorithm Minimax with static evaluator

    Inputs:
    Connect N board
    Depth- Minimax Learner will build tree of next possible moves to that depth
    N- number of tokens that need to be connected for a player to win
    Player- player number, either 1 or -1
    Algorithm- either "minimax" or "ab" for alpha beta pruned minimax
    """

    def __init__(self, board, depth, n, player, alg):
        self.board = board
        self.depth = depth
        self.num_states = board.grid_size
        self.player = player
        self.n = n
        self.alg = alg


    """
    Following streak functions check if player has token streak in the four possible win directions
    """
    def streakVertical(self, board, col, row, player):
        if row > len(board[col]) - self.n:
            return 0
        for i in range(0,self.n):
            if board[col][row + i] == -1*player:
                return 0
            if board[col][row + i] == 0:
                return i
        return self.n

    def streakHorizontal(self, board, col, row, player):
        if col > len(board) - self.n:
            return 0
        for i in range(0,self.n):
            if board[col + i][row] == -1*player:
                return 0
            if board[col + i][row] == 0:
                return i
        return self.n

    def streakDiagonalUp(self, board, col, row, player):
        if row > len(board[col]) - self.n or col > len(board) - self.n:
            return 0
        for i in range(0,self.n):
            if board[col + i][row + i] == -1*player:
                return 0
            if board[col + i][row + i] == 0:
                return i
        return self.n

    def streakDiagonalDown(self, board, col, row, player):
        if row < self.n or col > len(board) - self.n:
            return 0
        for i in range(0,self.n):
            if board[col + i][row - i] == -1*player:
                return 0
            if board[col + i][row - i] == 0:
                return i
        return self.n

    def value(self, board):
        """
        Calculates value of board states
        """
        val = 0
        conversion = [int(math.pow(2, i))/2 for i in range(0, self.n+1)]
        conversion[self.n] = 20000000
        conversion_other = [int(math.pow(2, i))/4 for i in range(0, self.n+1)]
        conversion_other[self.n] = 1000000
        for i in range(0, len(board)):
            for j in range(0, len(board[0])):
                temp = self.streakVertical(board, i, j, self.player)
                if temp == self.n:
                    return conversion[temp]
                val += conversion[temp]
                temp = self.streakHorizontal(board, i, j, self.player)

                if temp == self.n:
                    return conversion[temp]
                val += conversion[temp]
                temp = self.streakDiagonalUp(board, i, j, self.player)
                if temp == self.n:
                    return conversion[temp]
                val += conversion[temp]

                temp = self.streakDiagonalDown(board, i, j, self.player)
                if temp == self.n:
                    return conversion[temp]
                val += conversion[temp]


                temp = self.streakVertical(board, i, j, -1*self.player)
                if temp == self.n:
                    return -1*conversion_other[temp]
                val -= conversion[temp]

                temp = self.streakHorizontal(board, i, j, -1*self.player)
                if temp == self.n:
                    return -1*conversion_other[temp]
                val -= conversion[temp]
                temp = self.streakDiagonalUp(board, i, j, -1*self.player)
                if temp == self.n:
                    return -1*conversion_other[temp]
                val -= conversion[temp]
                temp = self.streakDiagonalDown(board, i, j, -1*self.player)
                if temp == self.n:
                    return -1*conversion_other[temp]
                val -= conversion[temp]

        return val

    def create_tree(self, node, depth, player, move):
        """
        Creates tree of next possible moves

        Each node is a dict of node value, children, the board state, which player's turn it would be, and move
        """
        if depth == 0:
            return None

        else:
            tree = {"value": 0, "children": [], "board": node, "player": player, "move": move}

            next_moves = node.next_possible_moves()

            for move in next_moves:
                board_copy = copy.deepcopy(node)
                board_copy.move(move, player)
                new_child = self.create_tree(board_copy, depth-1, -1*player, move)

                if new_child != None:
                    tree["children"].append(new_child)

            return tree

    def children(self, node):
        """
        returns children of a node
        """
        return node["children"]

    def leaf(self, node):
        """
        returns if current node is a leaf (i.e. no children)
        """
        return len(self.children(node)) == 0

    def max_node(self, node):
        """
        returns true if node is a max node
        """
        return node["player"] == self.player

    def evaluate(self, node):
        """
        Static evaluator function to return a value between Loss and Win for intermediate game
        positions, larger if the position is better for the current player.
        If depth limit of the search is exceeded, is applied to remaining nodes as if
        they were leaves.

        We calculate the rating by checking each token already placed, and
        checking how many possible ways to connect N there are
        """
        node["value"] = self.value(node["board"].grid)
        return node["value"]

    def minimax(self, node, depth):
        """
        Recursive implementation of Minimax algorithm using pseudocode from:
        https://www.cs.cornell.edu/courses/cs312/2002sp/lectures/rec21.htm
        """
        if self.leaf(node) or depth == 0:
            return self.evaluate(node)

        if self.max_node(node):
            # L = -infinity
            current_node_value = -1000000000
            for child in self.children(node):
                next_node_value = self.minimax(child, depth-1)
                if current_node_value < next_node_value:
                    current_node_value = next_node_value
            node["value"] = current_node_value
            return current_node_value

        if not self.max_node(node):
            # W = +infinity
            current_node_value = 10000000000
            for child in self.children(node):
                next_node_value = self.minimax(child, depth-1)
                if next_node_value < current_node_value:
                    current_node_value = next_node_value
            node["value"] = current_node_value
            return current_node_value


    def ab_minimax(self, node, depth, min_val, max_val):
        """
        Implementation of Minimax with Alpha Beta Pruning

        In contrast to previous minimax algorithm, must now input min_val and max_val as well
        """
        if self.leaf(node) or depth == 0:
            return self.evaluate(node)

        if self.max_node(node):
            current_node_value = min_val
            for child in self.children(node):
                next_node_value = self.ab_minimax(child, depth-1, current_node_value, max_val)
                if current_node_value < next_node_value:
                    current_node_value = next_node_value
                if current_node_value > max_val:
                    return max_val
            node["value"] = current_node_value
            return current_node_value

        if not self.max_node(node):
            current_node_value = max_val
            for child in self.children(node):
                next_node_value = self.ab_minimax(child, depth-1, min_val, current_node_value)
                if next_node_value < current_node_value:
                    current_node_value = next_node_value
                if current_node_value < min_val:
                    return min_val
            node["value"] = current_node_value
            return current_node_value

    def calc_next_move(self):
        """
        Calculate Minimax's Learners optimal next move
        """
        current_tree = self.create_tree(self.board, self.depth, self.player, None)
        self.board.print_grid()

        if self.alg == "minimax":
            top_val = self.minimax(current_tree, self.depth)
        elif self.alg == "ab":
            top_val = self.ab_minimax(current_tree, self.depth, -100000, 100000)

        print "this is top_val", top_val

        for child in current_tree["children"]:
            if child["value"] == top_val:
                return child["move"]

        top_val = np.min([x["value"] for x in current_tree["children"]])
        for child in current_tree["children"]:
            if child["value"] == top_val:
                return child["move"]
