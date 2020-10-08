import random
import numpy as np
import yaml

from neural_network import Neural_Network


file = open('config.yml', 'r')
cfg = yaml.load(file, Loader=yaml.FullLoader)

class Vehicle:
    """
    Vehicle object for Car ML Simulator.
    Attributes:
    - car_id
    - x
    - y
    - speed
    - model
    - training_data_assigned
    - training_label_assigned
    - gradients
    """
    def __init__(self, car_id):
        self.car_id = car_id
        self.x = 0
        self.y = 0
        self.speed = 0
        self.model = None
        self.training_data_assigned = []
        self.training_label_assigned = []                     
        self.gradients = None
        # self.rsu_assigned = None


    def set_properties(self, x, y, speed):
        self.x = x
        self.y = y
        self.speed = speed

    def download_model_from(self, central_server):
        self.model = central_server.model

    def compute(self, simulation):
        neural_net = Neural_Network()
        self.gradients = neural_net.grad(self.model, np.array(self.training_data_assigned), np.array(self.training_label_assigned), simulation)

    def upload(self, simulation):
        rsu = random.choice(simulation.rsu_list)
        rsu.accumulative_gradients.append(self.gradients)
        # RSU checks if enough gradients collected
        if len(rsu.accumulative_gradients) >= cfg['simulation']['maximum_rsu_accumulative_gradients']:
            rsu.communicate_with_central_server(simulation.central_server)

    def compute_and_upload(self, simulation):
        self.compute(simulation)
        self.upload(simulation)


    
    # # Return the RSU that is cloest to the vehicle
    # def closest_rsu(self, rsu_list):
    #     shortest_distance = 99999999 # placeholder (a random large number)
    #     closest_rsu = None
    #     for rsu in rsu_list:
    #         distance = math.sqrt((rsu.rsu_x - self.x) ** 2 + (rsu.rsu_y - self.y) ** 2)
    #         if distance <= rsu.rsu_range and distance < shortest_distance:
    #             shortest_distance = distance
    #             closest_rsu = rsu
    #     return closest_rsu

    # # Return a list of RSUs that is within the range of the vehicle
    # # with each RSU being sorted from the closest to the furtherst
    # def in_range_rsus(self, rsu_list):
    #     in_range_rsus = []
    #     for rsu in rsu_list:
    #         distance = math.sqrt((rsu.rsu_x - self.x) ** 2 + (rsu.rsu_y - self.y) ** 2)
    #         if distance <= rsu.rsu_range:
    #             heapq.heappush(in_range_rsus, (distance, rsu))
    #     return [heapq.heappop(in_range_rsus)[1] for i in range(len(in_range_rsus))]
