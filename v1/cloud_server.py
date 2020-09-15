import yaml
import tcp_server


# Load configuration file
file = open('config.yml', 'r')
cfg = yaml.load(file, Loader=yaml.FullLoader)


class Cloud_Server:
    """
    Cloud Server object.
    Attributes:
    - rsu_list
    - model
    """

    def __init__(self, model):
        # self.rsu_list = rsu_list
        self.model = model

    def run_cloud_server(self):
        host = cfg['cloud_server']['host']
        cloud_server = tcp_server.Server(host)
        cloud_server.run_server()
        # After server ran, update ip and port
        self.ip = cloud_server.ip
        self.port = cloud_server.port
        print("Cloud server starts on HOST: {}, PORT: {}".format(cloud_server.ip, 
                                                                 cloud_server.port))



# Code for testing
if __name__ == "__main__":
    c = Cloud_Server([])
    c.run_cloud_server()