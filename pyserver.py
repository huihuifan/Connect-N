import threading
import webbrowser
import BaseHTTPServer
import SimpleHTTPServer
import urlparse
import json
import numpy as np

import learners

PORT = 8000


class TestHandler(SimpleHTTPServer.SimpleHTTPRequestHandler):
    """The test example handler."""

    def do_GET(self):
        ind = self.path.find("stuff=")
        if (ind == -1):
            SimpleHTTPServer.SimpleHTTPRequestHandler.do_GET(self)
        else:
            arr = json.loads(self.path[ind + 6:])
            x = learners.ConnectN(7,4)
            x.grid = np.array(arr)
            p1 = learners.Minimax_Learner(x, 3, 4, -1, "minimax")
            self.wfile.write(p1.calc_next_move())



def start_server():
    """Start the server."""
    server_address = ("", PORT)
    server = BaseHTTPServer.HTTPServer(server_address, TestHandler)
    server.serve_forever()

if __name__ == "__main__":
    start_server()
