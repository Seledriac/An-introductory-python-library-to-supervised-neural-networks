
# -*-coding:utf-8 -*


"""This is a testing script for the neural network module"""

#Network module
import network
#Numpy is useful to test the Network 
import numpy as np
#Pickle is useful to save the network's state
import pickle

#Tests

#Creating a randomly initialized network of 10 input neurons and 10 output neurons
net = network.Network("net", [10,10])
#We test the output of this network when given a column matrix of the 10 digits
print("\nOutput of the created Network \"{0}\" when given a \"np.arange(10).reshape(10, 1)\" input : \n{1}".format(net, net.feedforward(np.arange(10).reshape(10, 1))))

#Saving it into the "serialized_network" file
with open("serialized_test_network", "wb") as saving:
    saver = pickle.Pickler(saving)
    saver.dump(net)

#Deserialization of the saved network
deserialized_network = None
with open("serialized_test_network", "rb") as retrieving_model:
    retriever = pickle.Unpickler(retrieving_model)
    deserialized_network = retriever.load()

#Verification that the model was successfully saved
print("\nDeserialized network : \"{0}\"".format(deserialized_network))
print("\nOutput of the deserialized Network \"{0}\" when given a \"np.arange(10).reshape(10, 1)\" input : \n{1}".format(net, net.feedforward(np.arange(10).reshape(10, 1))))
print("Hence, the Network Model was successfully saved !\n")


