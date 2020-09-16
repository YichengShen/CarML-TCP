import socket
import sys
import yaml

import tcp_client


# Load configuration file
file = open('config.yml', 'r')
cfg = yaml.load(file, Loader=yaml.FullLoader)


class Vehicle:
    def __init__(self):
        self.model = 0

    def run_as_client(self):
        rsu_id = int(sys.argv[1])	
        if rsu_id == 1:
            PORT = cfg['rsu']['port1']
        elif rsu_id == 2:
            PORT = cfg['rsu']['port2']
        else:
            PORT = cfg['rsu']['port3']
        self.model = tcp_client.Client.run_client(PORT, msg=1)
        

if __name__ == "__main__":
    v = Vehicle()
    v.run_as_client()