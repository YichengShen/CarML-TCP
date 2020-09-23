from neural_network import Neural_Network 
import numpy as np
import yaml

file = open('config.yml', 'r')
cfg = yaml.load(file, Loader=yaml.FullLoader)

class RSU:
    """
    Road Side Unit object for Car ML Simulator.
    Attributes:
    - rsu_id
    - rsu_x
    - rsu_y
    - rsu_range
    - accumulative_gradients
    """
    def __init__(self, rsu_id, rsu_x, rsu_y, rsu_range, traffic_proportion):
        self.rsu_id = rsu_id
        self.rsu_x = rsu_x
        self.rsu_y = rsu_y
        self.rsu_range = rsu_range
        self.accumulative_gradients = []

    def aggregate(self, central_server):
        neural_network = Neural_Network()
        grads, self.accumulative_gradients = self.accumulative_gradients[:10], self.accumulative_gradients[10:]
        #CGC

        accumulative_gradients = neural_network.accumulate_gradients_itr(grads)
        accumulative_gradients = np.true_divide(accumulative_gradients, len(grads))
        central_server.accumulative_gradients.append(accumulative_gradients)

    # The RSU updates the model in the central server with its accumulative gradients and downloads the 
    # latest model from the central server
    def communicate_with_central_server(self, central_server):
        self.aggregate(central_server)
        if len(central_server.accumulative_gradients) >= cfg['simulation']['maximum_rsu_accumulative_gradients']:
            central_server.update_model()
