import yaml
import socket
from _thread import *
import sys

# import tcp_server


# Load configuration file
file = open('config.yml', 'r')
cfg = yaml.load(file, Loader=yaml.FullLoader)


class Cloud_Server:
    """
    Cloud Server object.
    Attributes:
    - model
    """

    def __init__(self):
        self.model = 0

    def run_cloud_server(self):
        HOST = socket.gethostname()	# Symbolic name meaning all available interfaces
        PORT = cfg['cloud_server']['port']

        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        print('Socket created')

        #Bind socket to local host and port
        try:
            s.bind((HOST, PORT))
        except socket.error as msg:
            print('Bind failed. Error :', msg)
            sys.exit()
	
        print('Socket bind complete')

        #Start listening on socket
        s.listen(10)
        print('Socket now listening')
        print("""Cloud server running on\n ---> HOST: {}, PORT: {}""".format(socket.gethostbyname(HOST), 
                                                   PORT))

        #Function for handling connections. This will be used to create threads
        def clientthread(conn):
            #infinite loop so that function do not terminate and thread do not end.
            while True:
                #Receiving from client
                data = int.from_bytes(conn.recv(1024), byteorder='big')
                if not data: 
                    break

                self.model += data
                print('current model : ', self.model)
            
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
        


# Run Cloud Server
if __name__ == "__main__":
    c = Cloud_Server()
    c.run_cloud_server()