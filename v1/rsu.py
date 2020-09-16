import yaml
import socket
import sys	
from _thread import *

import tcp_server, tcp_client


# Load configuration file
file = open('config.yml', 'r')
cfg = yaml.load(file, Loader=yaml.FullLoader)


class RSU:
    """
    RSU object.
    Attributes:
    - id
    - model
    - gradient
    """

    def __init__(self, id):
        self.id = id
        self.model = 0
        self.gradient = 0
 
    def run_as_client(self, port):
        model = tcp_client.Client.run_client(port, msg=self.gradient)
        return model

    def run_as_server(self):
        HOST = socket.gethostname()	# Symbolic name meaning all available interfaces
        rsu_id = int(sys.argv[1])	
        if rsu_id == 1:
            PORT = cfg['rsu']['port1']
        elif rsu_id == 2:
            PORT = cfg['rsu']['port2']
        else:
            PORT = cfg['rsu']['port3']

        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        print('Socket created')

        #Bind socket to local host and port
        try:
            s.bind((HOST, PORT))
        except socket.error as msg:
            print('Bind failed. Error :', msg)
            sys.exit()
        print(PORT)
        print('Socket bind complete')

        #Start listening on socket
        s.listen(10)
        print('Socket now listening')
        print("""RSU running on\n ---> HOST: {}, PORT: {}""".format(socket.gethostbyname(HOST), 
                                                   PORT))

        #Function for handling connections. This will be used to create threads
        def clientthread(conn):
            #infinite loop so that function do not terminate and thread do not end.
            while True:
                #Receiving from client
                data = int.from_bytes(conn.recv(1024), byteorder='big')
                if not data: 
                    break

                self.gradient += data
                print('gradient : ', self.gradient)

                self.model = self.run_as_client(8888)
            
                conn.sendall(self.model.to_bytes(2, 'big'))
            #came out of loop
            conn.close()

        #now keep talking with the client
        while True:
            #wait to accept a connection - blocking call
            conn, addr = s.accept()
            print('Connected with ' + addr[0] + ':' + str(addr[1]))
            
            #start new thread takes 1st argument as a function name to be run, second is the tuple of arguments to the function.
            start_new_thread(clientthread, (conn,))

        s.close()
        

    
# Code for testing
if __name__ == "__main__":
    testing_rsu1 = RSU(1)
    testing_rsu1.run_as_server()