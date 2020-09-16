import socket	
import sys	


class Client:
    """
    TCP Client containing the following functions:
    - run_client
    """

    def run_client(port, msg):
        #create an INET, STREAMing socket
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        except socket.error:
            print('Failed to create socket')
            sys.exit()
            
        print('Socket Created')

        host = socket.gethostname()
        # port = passed in parameter

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
        message = msg # for rsu -> cloud, msg is gradient

        try :
            #Set the whole string
            s.sendall((message).to_bytes(2, 'big'))
        except socket.error:
            #Send failed
            print('Send failed')
            sys.exit()

        print('Message send successfully')

        #Now receive data
        model = int.from_bytes(s.recv(4096), byteorder='big')

        print('model returned : ', model)

        s.close()

        return model


# Code for testing
if __name__ == "__main__":
    HOST, PORT = "localhost", 9999

    testing_client1 = Client(HOST, PORT)
    testing_client1.connect()

    testing_client2 = Client(HOST, PORT)
    testing_client2.connect()