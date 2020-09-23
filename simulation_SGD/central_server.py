from neural_network import Neural_Network
from vehicle import Vehicle
import numpy as np
import tensorflow as tf
import yaml

file = open('config.yml', 'r')
cfg = yaml.load(file, Loader=yaml.FullLoader)

class Central_Server:
    """
    Central Server object for Car ML Simulator.
    Attributes:
    - model
    - accumulative_gradients
    """
    def __init__(self, rsu_list):
        # The structure of the neural network
        self.model = tf.keras.Sequential([
                    tf.keras.layers.Dense(128, activation=tf.nn.relu, input_shape=(3, 1024,)),  # input shape required
                    # tf.keras.layers.Dense(10, activation=tf.nn.relu),
                    tf.keras.layers.Dense(10)
        ])
        self.accumulative_gradients = []

    # Update the model with its accumulative gradients
    # Used for batch gradient descent
    def update_model(self):
        if len(self.accumulative_gradients) >= 10:
            neural_network = Neural_Network()
            grads, self.accumulative_gradients = self.accumulative_gradients[:10], self.accumulative_gradients[10:]
            #CGC
            accumulative_gradients = neural_network.accumulate_gradients_itr(grads)
            accumulative_gradients = np.true_divide(self.accumulative_gradients, len(grads))
            gradient_zip = zip(accumulative_gradients, self.model.trainable_variables)
            neural_network.optimizer.apply_gradients(gradient_zip)


class Simulation:
    """
    Simulation object for Car ML Simulator. Stores all the global variables.
    Attributes:
    - FCD_file
    - vehicle_dict
    - rsu_list
    - dataset
    """
    def __init__(self, FCD_file, vehicle_dict: dict, rsu_list: list, central_server, training_set):
        self.FCD_file = FCD_file
        self.vehicle_dict = vehicle_dict
        self.rsu_list = rsu_list
        self.central_server = central_server
        self.num_epoch = 0
        self.training_data = []
        self.epoch_loss_avg = None
        self.epoch_accuracy = None
        self.training_set = training_set
       
    def add_into_vehicle_dict(self, vehicle):
        self.vehicle_dict[vehicle.attrib['id']] = Vehicle(vehicle.attrib['id'])

    def print_accuracy(self):
        print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(self.num_epoch,
                                                                self.epoch_loss_avg.result(),
                                                                self.epoch_accuracy.result()))

    def new_epoch(self):
        self.num_epoch += 1
        for data, label in self.training_set.as_numpy_iterator():
            self.training_data.append((data.tolist(), label.tolist()))

