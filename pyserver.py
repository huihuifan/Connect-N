import threading
import webbrowser
import BaseHTTPServer
import SimpleHTTPServer
import urlparse
import json
import numpy as np
import sys

from agents import Q_Learner, Minimax_Learner, MCTS
from simulator import ConnectN

Q_value_table = None
last_action = -1
last_board = None

class TestHandler(SimpleHTTPServer.SimpleHTTPRequestHandler):
    """The test example handler."""

    def do_GET(self):
        global Q_value_table
        global last_action
        global last_board
        ind = self.path.find("stuff=")
        ind2 = self.path.find("&col=")
        ind3 = self.path.find("&agent=")

        agent = self.path[ind3 + 7:]

        if (ind == -1):
            SimpleHTTPServer.SimpleHTTPRequestHandler.do_GET(self)
        else:
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            arr = json.loads(self.path[ind + 6:ind2])
            x = ConnectN(7,4)
            x.grid = np.array(arr)
            last_board = x
            col = int(self.path[ind2 + 5:ind3])
            if (x.check_win(col, np.sum([abs(j) for j in x.grid[col]]) - 1, 1) == 1):
                if agent == "QL":
                    p1 = Q_Learner(x, Q_value_table, True)
                    p1.last_board_state = last_board.grid
                    p1.last_action = last_action
                    p1.calc_next_move(-50, x)
                self.wfile.write(-100)
            else:
                if agent == "Minimax":
                    p1 = Minimax_Learner(x, 3, 4, -1, "minimax")
                    next_move = p1.calc_next_move()
                elif agent == "QL":
                    p1 = Q_Learner(x, Q_value_table, True)
                    next_move = p1.calc_next_move(None, x)
                    last_action = next_move
                    Q_value_table = p1.value_table
                else:
                    p1 = MCTS(x, 1000, 0.5)
                    next_move = p1.calc_next_move()

                x.move(next_move, -1)
                if (x.check_win(next_move, np.sum([abs(j) for j in x.grid[next_move]]) - 1, -1) == 1):
                    if agent == "QL":
                        p1.calc_next_move(50, x)
                    self.wfile.write(-1*next_move)
                else:
                    self.wfile.write(next_move)



def start_server():
    """Start the server."""
    server_address = ("", int(sys.argv[1]))
    server = BaseHTTPServer.HTTPServer(server_address, TestHandler)
    server.serve_forever()

if __name__ == "__main__":
    start_server()
