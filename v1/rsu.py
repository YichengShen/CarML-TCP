import yaml
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
    """

    def __init__(self, id, cloud_ip, cloud_port):
        self.id = id
        self.cloud_ip = cloud_ip
        self.cloud_port = cloud_port
        # self.model =

    def run_rsu_as_client(self):
        rsu_client = tcp_client.Client(self.cloud_ip, self.cloud_port)
        rsu_client.connect()

    def run_rsu_as_server(self):
        pass

    
# Code for testing
if __name__ == "__main__":
    testing_rsu1 = RSU(1)
    testing_rsu1.run_rsu_as_client()