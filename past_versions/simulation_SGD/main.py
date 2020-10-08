from sumo import SUMO_Dataset
from central_server import Central_Server, Simulation
from vehicle import Vehicle

import random
import yaml
# from locationPicker_v3 import output_junctions
import xml.etree.ElementTree as ET 
import tensorflow as tf
import threading


file = open('config.yml', 'r')
cfg = yaml.load(file, Loader=yaml.FullLoader)

def simulate(simulation):
    tree = ET.parse(simulation.FCD_file)
    root = tree.getroot()
    simulation.new_epoch()
    
    # Maximum training epochs
    while simulation.num_epoch <= cfg['neural_network']['epoch']:
        # For each time step (sec) in the FCD file 
        for timestep in root:

            if simulation.num_epoch > cfg['neural_network']['epoch']:
                break

            # For each vehicle on the map at the timestep
            for vehicle in timestep.findall('vehicle'):
                # If vehicle not yet stored in vehicle_dict
                if vehicle.attrib['id'] not in simulation.vehicle_dict:
                    simulation.add_into_vehicle_dict(vehicle)
                # Get the vehicle object from vehicle_dict
                vehi = simulation.vehicle_dict[vehicle.attrib['id']]  
                # Set location and speed
                vehi.set_properties(float(vehicle.attrib['x']),
                                    float(vehicle.attrib['y']),
                                    float(vehicle.attrib['speed']))

                # Download Training Data / New Epoch
                if simulation.training_data:
                    vehi.training_data_assigned, vehi.training_label_assigned = simulation.training_data.pop()
                else:
                    simulation.print_accuracy()
                    simulation.new_epoch()
                    vehi.training_data_assigned, vehi.training_label_assigned = simulation.training_data.pop()
                
                # Download Model
                vehi.download_model_from(simulation.central_server)

                x = threading.Thread(target=vehi.compute_and_upload(simulation))
                x.start()
                
    return simulation.central_server.model


def main():
    ROU_FILE = cfg['simulation']['ROU_FILE']
    NET_FILE = cfg['simulation']['NET_FILE']
    FCD_FILE = cfg['simulation']['FCD_FILE']
    
    RSU_RANGE = cfg['comm_range']['v2rsu']           # range of RSU
    NUM_RSU = cfg['simulation']['num_rsu']           # number of RSU

    sumo_data = SUMO_Dataset(ROU_FILE, NET_FILE)
    vehicle_dict = {}
    # rsu_list = sumo_data.rsuList(RSU_RANGE, NUM_RSU, output_junctions)
    rsu_list = sumo_data.rsuList_random(RSU_RANGE, NUM_RSU)
    central_server = Central_Server(rsu_list)

    # Load Data
    train, test = tf.keras.datasets.cifar10.load_data()

    # Normalize the training data to fit the model
    train_images, train_labels = train
    num_training_data = cfg['simulation']['num_training_data']
    train_images, train_labels = train_images[:num_training_data], train_labels[:num_training_data]
    train_images = train_images.reshape(train_images.shape[0], 32, 32, 3)
    train_images = train_images/255

    # Normalize the testing data to fit the model
    test_images, test_labels = test
    test_images, test_labels = test_images, test_labels
    test_images = test_images.reshape(test_images.shape[0], 32, 32, 3)
    test_images = test_images/255

    batch_size = cfg['neural_network']['batch_size']
    training_set = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(int(num_training_data/batch_size)).batch(batch_size)
    test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(batch_size)

    simulation = Simulation(FCD_FILE, vehicle_dict, rsu_list, central_server, training_set)
    model = simulate(simulation)

    # Test the accuracy of the computed model
    test_accuracy = tf.keras.metrics.Accuracy()     
    for (x, y) in test_dataset:
        logits = model(x, training=False) 
        prediction = tf.argmax(logits, axis=1, output_type=tf.int32)
        test_accuracy(prediction, y)
    print("Test set accuracy: {:.3%}".format(test_accuracy.result()))


if __name__ == '__main__':
    main()
