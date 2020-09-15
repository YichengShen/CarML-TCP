import socket


class Client:
    """
    TCP Client object.
    Attributes:
    - ip
    - port
    """

    def __init__(self, ip, port):
        self.ip = ip
        self.port = port

    def connect(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.connect((self.ip, self.port))
            sock.sendall(bytes(message, 'ascii'))
            response = str(sock.recv(1024), 'ascii')
            print("Received: {}".format(response))


# Code for testing
if __name__ == "__main__":
    HOST, PORT = "localhost", 9999

    testing_client1 = Client(HOST, PORT)
    testing_client1.connect()

    testing_client2 = Client(HOST, PORT)
    testing_client2.connect()