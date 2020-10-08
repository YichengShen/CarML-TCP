from neural_network import Neural_Network
from vehicle import Vehicle
import numpy as np
import tensorflow as tf
import yaml

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, Activation, MaxPooling2D
from tensorflow.keras.layers import AveragePooling2D, Input, Flatten, Dropout, BatchNormalization



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
        # nb_classes = 10
        # img_rows, img_col = 32, 32
        # img_channels = 3
        # nb_filters = 32
        # nb_pool = 2
        # nb_conv = 3
        # # The structure of the neural network
        self.model = Sequential()

        # self.model.add(Conv2D(nb_filters, (nb_conv, nb_conv),
        #                 padding='valid',
        #                 activation='relu',
        #                 input_shape=(img_rows, img_col, img_channels),
        #                 data_format='channels_last',))

        # self.model.add(Conv2D(nb_filters, (nb_conv, nb_conv), activation='relu'))
        # self.model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
        # self.model.add(Dropout(0.5))

        # self.model.add(Conv2D(nb_filters, (nb_conv, nb_conv), activation='relu'))
        # self.model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
        # self.model.add(Dropout(0.5))

        # self.model.add(Flatten())
        # self.model.add(Dense(128, activation='relu'))
        # self.model.add(Dropout(0.5))
        # self.model.add(Dense(nb_classes, activation='softmax'))

        # self.model.compile(loss='categorical_crossentropy',
        #             optimizer='adam',
        #             metrics=['accuracy'])

        # num_filters = 64
        # #  First convolutional layer
        # self.model.add(Conv2D(filters=num_filters, kernel_size=3, padding="SAME", activation='relu'))
        # self.model.add(BatchNormalization())
        # self.model.add(Conv2D(filters=num_filters, kernel_size=3, padding="SAME", activation='relu'))
        # self.model.add(BatchNormalization())
        # self.model.add(MaxPooling2D(pool_size=2, strides=2))
        # self.model.add(Dropout(rate=0.25))
        # #  Second convolutional layer
        # # self.model.add(MaxPool2D(pool_size=2, strides=2))
        # # Third convolutional layer
        # self.model.add(Conv2D(filters=num_filters, kernel_size=3, padding="SAME", activation='relu'))
        # self.model.add(BatchNormalization())
        # self.model.add(Conv2D(filters=num_filters, kernel_size=3, padding="SAME", activation='relu'))
        # self.model.add(BatchNormalization())
        # self.model.add(MaxPooling2D(pool_size=2, strides=2))
        # self.model.add(Dropout(rate=0.25))
        # # self.model.add(Conv2D(channels=64, kernel_size=3, padding=(1,1), activation='relu'))
        # # self.model.add(Conv2D(channels=64, kernel_size=3, padding=(1,1), activation='relu'))
        # # self.model.add(Conv2D(channels=64, kernel_size=3, padding=(1,1), activation='relu'))
        # # self.model.add(MaxPool2D(pool_size=2, strides=2))
        # # Flatten and apply fullly connected layers
        # self.model.add(Flatten())
        # # self.model.add(Dense(512, activation="relu"))
        # # self.model.add(Dense(512, activation="relu"))
        # self.model.add(Dense(128, activation="relu"))
        # # self.model.add(Dense(256, activation="relu"))
        # self.model.add(Dropout(rate=0.25))
        # self.model.add(Dense(10))
        
        selfmodel = tf.keras.Sequential([
                    tf.keras.layers.Dense(128, activation=tf.nn.relu, input_shape=(32, 32, 3)),  # input shape required
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
    def __init__(self, FCD_file, vehicle_dict: dict, rsu_list: list, central_server, training_set):
        self.FCD_file = FCD_file
        self.vehicle_dict = vehicle_dict
        self.rsu_list = rsu_list
        self.central_server = central_server
        self.num_epoch = 0
        self.training_data = []
        self.epoch_loss_avg = tf.keras.metrics.Mean()
        self.epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
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

