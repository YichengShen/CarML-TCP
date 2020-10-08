#Socket client example in python

import socket	#for sockets
import sys	#for exit

class Vehicle:
    def __init__(self):
        self.model = 0

    def run_client(self):
        #create an INET, STREAMing socket
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        except socket.error:
            print('Failed to create socket')
            sys.exit()
            
        print('Socket Created')

        host = socket.gethostname()
        port = 7777

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
        message = 1

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

        print('model received : ', self.model)

        s.close()

if __name__ == "__main__":
    rsu = Vehicle()
    rsu.run_client()