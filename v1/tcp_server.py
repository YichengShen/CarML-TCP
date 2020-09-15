import socketserver
import threading


class ThreadedTCPRequestHandler(socketserver.BaseRequestHandler):

    def handle(self):
        data = str(self.request.recv(1024), 'ascii')
        cur_thread = threading.current_thread()
        response = bytes("{}: {}".format(cur_thread.name, data), 'ascii')
        self.request.sendall(response)

class ThreadedTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    pass


class Server:
    """
    TCP Server object.
    Attributes:
    - host
    - port
    """

    def __init__(self, host):
        self.host = host
        self.ip = host
        self.port = 0 # arbitrary unused port

    def run_server(self):
        server = ThreadedTCPServer((self.host, self.port), ThreadedTCPRequestHandler)
        with server:
            self.ip, self.port = server.server_address

            # Start a thread with the server -- that thread will then start one
            # more thread for each request
            server_thread = threading.Thread(target=server.serve_forever)
            # Exit the server thread when the main thread terminates
            server_thread.daemon = True
            server_thread.start()
            print("Server loop running in thread:", server_thread.name)



# Code for testing
if __name__ == "__main__":
    HOST = "localhost"

    testing_server = Server(HOST)
    testing_server.run_server()
    