import socket	#for sockets
import sys	#for exit
from _thread import *

class RSU:
    def __init__(self):
        self.model = 0
        self.gradient = 0

    def run_server(self):
        HOST = socket.gethostname()	# Symbolic name meaning all available interfaces
        PORT = int(sys.argv[1])	# Arbitrary non-privileged port

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

                self.run_client()
            
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

    def run_client(self):
        #create an INET, STREAMing socket
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        except socket.error:
            print('Failed to create socket')
            sys.exit()
            
        print('Socket Created')

        host = socket.gethostname()
        port = 9999

        try:
            remote_ip = socket.gethostbyname(host)

        except socket.gaierror:
            #could not resolve
            print('Hostname could not be resolved. Exiting')
            sys.exit()

        #Connect to remote server
        s.connect((remote_ip , port))

        print('Socket Connected to ' + host + ' on ip ' + remote_ip)

        #Send some data to remote server
        message = self.gradient

        try :
            #Set the whole string
            s.sendall((message).to_bytes(2, 'big'))
        except socket.error:
            #Send failed
            print('Send failed')
            sys.exit()

        print('Message send successfully')

        #Now receive data
        self.model = int.from_bytes(s.recv(4096), byteorder='big')

        print('model returned : ', self.model)

        s.close()

if __name__ == "__main__":
    rsu = RSU()

    rsu.run_server()