from neural_network import Neural_Network
from vehicle import Vehicle
import numpy as np
import yaml
import mxnet as mx
from mxnet import gluon, nd



file = open('config.yml', 'r')
cfg = yaml.load(file, Loader=yaml.FullLoader)

class Central_Server:
    """
    Central Server object for Car ML Simulator.
    Attributes:
    - model
    - accumulative_gradients
    """
    def __init__(self, ctx, rsu_list):
        self.net = gluon.nn.Sequential()
        with self.net.name_scope():
            #  First convolutional layer
            self.net.add(gluon.nn.Conv2D(channels=64, kernel_size=3, padding=(1,1), activation='relu'))
            self.net.add(gluon.nn.BatchNorm())
            self.net.add(gluon.nn.Conv2D(channels=64, kernel_size=3, padding=(1,1), activation='relu'))
            self.net.add(gluon.nn.BatchNorm())
            self.net.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))
            self.net.add(gluon.nn.Dropout(rate=0.25))
            #  Second convolutional layer
            # net.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))
            # Third convolutional layer
            self.net.add(gluon.nn.Conv2D(channels=128, kernel_size=3, padding=(1,1), activation='relu'))
            self.net.add(gluon.nn.BatchNorm())
            self.net.add(gluon.nn.Conv2D(channels=128, kernel_size=3, padding=(1,1), activation='relu'))
            self.net.add(gluon.nn.BatchNorm())
            self.net.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))
            self.net.add(gluon.nn.Dropout(rate=0.25))
            # net.add(gluon.nn.Conv2D(channels=64, kernel_size=3, padding=(1,1), activation='relu'))
            # net.add(gluon.nn.Conv2D(channels=64, kernel_size=3, padding=(1,1), activation='relu'))
            # net.add(gluon.nn.Conv2D(channels=64, kernel_size=3, padding=(1,1), activation='relu'))
            # net.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))
            # Flatten and apply fullly connected layers
            self.net.add(gluon.nn.Flatten())
            # net.add(gluon.nn.Dense(512, activation="relu"))
            # net.add(gluon.nn.Dense(512, activation="relu"))
            self.net.add(gluon.nn.Dense(128, activation="relu"))
            # net.add(gluon.nn.Dense(256, activation="relu"))
            self.net.add(gluon.nn.Dropout(rate=0.25))
            self.net.add(gluon.nn.Dense(10)) # classes = 10
        self.net.initialize(mx.init.Xavier(), ctx=ctx, force_reinit=True)

        self.accumulative_gradients = []

    # Update the model with its accumulative gradients
    # Used for batch gradient descent
    def update_model(self):
        if len(self.accumulative_gradients) >= 10:
            neural_network = Neural_Network()
            grads, self.accumulative_gradients = self.accumulative_gradients[:10], self.accumulative_gradients[10:]
            #CGC
            accumulative_gradients = neural_network.accumulate_gradients_itr(grads)
            # accumulative_gradients = np.true_divide(self.accumulative_gradients, len(grads))
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
    def __init__(self, FCD_file, vehicle_dict: dict, rsu_list: list, central_server, training_set, val_train_data, val_test_data):
        self.FCD_file = FCD_file
        self.vehicle_dict = vehicle_dict
        self.rsu_list = rsu_list
        self.central_server = central_server
        self.num_epoch = 0
        self.training_data = []
        self.epoch_loss = mx.metric.CrossEntropy()
        self.epoch_accuracy = mx.metric.Accuracy()
        self.training_set = training_set
        self.val_train_data = val_train_data
        self.val_test_data = val_test_data
       
    def add_into_vehicle_dict(self, vehicle):
        self.vehicle_dict[vehicle.attrib['id']] = Vehicle(vehicle.attrib['id'])

    def print_accuracy(self):
        print("start")
        self.epoch_accuracy.reset()
        self.epoch_loss.reset()
        # accuracy on testing data
        for i, (data, label) in enumerate(self.val_test_data):
            outputs = self.central_server.net(data)
            self.epoch_accuracy.update(label, outputs)
        # cross entropy on training data
        for i, (data, label) in enumerate(self.val_train_data):
            outputs = self.central_server.net(data)
            self.epoch_loss.update(label, nd.softmax(outputs))

        _, accu = self.epoch_accuracy.get()
        _, loss = self.epoch_loss.get()
        # loss = 0
        print("Epoch {:03d}: Loss: {:03f}, Accuracy: {:03f}".format(self.num_epoch,
                                                                    loss,
                                                                    accu))
                                                                
    def new_epoch(self):
        self.num_epoch += 1
        for i, (data, label) in enumerate(self.training_set):
            self.training_data.append((data, label))

