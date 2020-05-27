
# -*-coding:utf-8 -*


"""
This module contains a class which models a neural network
"""

#Third-party libraries
import numpy as np
import random
from matplotlib import pyplot as plt

class Network():
    """The Network class, modeling a neural network, trainable with standard SGD/backpropagation algorithm method"""
    def __init__(self, id, sizes):
        """
        Network instance constructor. It assigns four descriptives attributes : 
            - Two attributes modeling the shape of the network :
                - "sizes", A list containing the size of each layer of the network in order, with the first number and last number being the shape of the input and output layers respectively
                - "num_layers", The total number of layers,
            - Two attributes giving the characteristics of each layer (randomly initialized)
                - "biases", A list of column matrixes, each representing the biases of each neuron for each layer
                - "weights", A list of 2D-matrixes, each containing the weights of the synapses leading to each neuron in a given layer (one neuron = one matrix line, one synapse = one matrix column)
        And an identifier "id"
        For instance, to create a Network with an input layer of 784 neurons, a hidden layer of 15 neurons, and an output layer of 10 neurons, you should do : "net = Network("net", [784,15,10])"
        """
        #The shape of the network is saved in theese two attributes 
        self.sizes = sizes
        self.num_layers = len(sizes)
        #Creates one bias column matrix for each layer appart from the input one
        self.biases = [np.random.randn(x, 1) for x in sizes[1:]]
        #Creates one 2D-matrix for each layer appart from the input one
        #Note : The weights matrixes linking the n-th layer of size x to the (n+1)th layer of size y are matrixes of size y,x : this makes the W.A + B computation straightforward (where A is the input)
        self.weights = [np.random.randn(y, x) for x,y in zip(sizes[:-1], sizes[1:])]
        #identifier initialization
        self.id = str(id)
    def feedforward(self, a):
        """This method returns the output of the network, given : the input a in parameter (in a matrix column form), the network's shape, its activation function, weights, and biases"""
        for w,b in zip(self.weights, self.biases):
            a = sigmoid(np.dot(w, a) + b)
        return a
    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data = None):
        """
        The Stochastic Gradient Descent training method, trains the network to make it behave like the training data, with the number of epochs specified, mini-batch size, and learning-rate "eta" specified. 
        You can add "test-data" to test your Network's performance and track it after each epoch
        """
        training_data = list(training_data)
        n = len(training_data)
        if test_data:
            test_data = list(test_data)
            n_test = len(test_data)
            #Display of the neural network's performance before training
            print("\nBeginning of the standard SGD method. Accuracy of the model at this state (with random weights and biases) : {}%\n".format(100 * self.evaluate(test_data) / n_test))
            print("The network will be trained with :\n- {0} epochs,\n- a mini-batch size of {1},\n- a learning rate of eta = {2}\n".format(epochs, mini_batch_size, eta))
            #This variable will track the model's performance through the epochs
            accuracies = [100 * self.evaluate(test_data) / n_test]
        for i in range(epochs):
            #For each epoch, we shuffle the training data to get different mini batches randomly extracted from the training set
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
            #And then we calculate the gradient, and apply the descent for each mini-batch
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                #After each epoch, we save the accuracy at this state
                accuracy = 100 * self.evaluate(test_data) / n_test
                accuracies.append(accuracy)
                print("Epoch n°{0} completed. Accuracy of the model at this state : {1}%".format(i + 1, 100 * self.evaluate(test_data) / n_test))
            else:
                print("Epoch n°{0} completed.".format(i))
        if test_data:
            #Upon training completion, we plot the training process
            self.plot_accuracy_graph(mini_batch_size, eta, range(epochs), accuracies)
    def update_mini_batch(self, mini_batch, eta):
        """This method applies the SGD to each weight and bias of the network given a mini-batch of training examples"""
        #We sum the delta_nablas (gradients of the cost function with respect to each weight and bias in the network) over all the training examples in the mini-batch
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            #All the partial derivatives for each weight and each bias are calculated via backpropagation
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        #And we apply the gradient descent to each weight and bias of the network according to the learning rate
        self.biases = [b - eta * (nb / len(mini_batch)) for b, nb in zip(self.biases, nabla_b)]
        self.weights = [w - eta * (nw / len(mini_batch)) for w, nw in zip(self.weights, nabla_w)]
    def backprop(self, x, y):
        """
        This method returns the gradient of the quadratic cost function (a function representing the square of the distance between the wanted output and the actual network's output)
        The gradient is calculated with respect to each weight and each bias of each neuron in the network for a given training example x and its wanted output y (both are column matrixes).
        The returned gradient is a tuple of (the gradient with respect to the biases, the gradient with respect to the weights)
        The two parts of the returned gradient are lists of matrixes shaped respectively like the biases and weights matrixes of the network instance
        """
        #initialization of the two gradient parts 
        delta_nabla_b = [np.zeros(b.shape) for b in self.biases]
        delta_nabla_w = [np.zeros(w.shape) for w in self.weights]
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        # We need to have access to each input and each output of each neuron of the network. Hence, we:
        # - calculate each w.x+b = z at each layer and store them in zs as a list of column matrixes
        # - store sigmoid(z) in the activations list for each layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        #First, we calculate the gradient of the output layer
        #delta is the product of the partial derivative of the quadratic cost with respect to the activation and the partial derivative of the activation with respect to the input z=w.x+b
        delta = self.quadratic_cost_derivative(activations[-1], y) * sigmoid_derivative(zs[-1])
        #the partial derivative of the input z=w.x+b with respect to the biases is always 1. Hence, dCost/db = delta
        delta_nabla_b[-1] = delta
        #the partial derivative of the input z=w.x+b with respect to the weights is the sum of the outputs of the last layer. Hence, dCost/dw = delta * sum(activations_of_last_layer) -> matrix product
        delta_nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        #Then, for each layer from the lasts to the firsts, we backpropagate the partial derivatives
        for l in range(2, self.num_layers): #For each hidden layer
            #We retrieve the input activations z=w.x+b
            z = zs[-l]
            #And store the partial derivatives of the activation function with respect to the activation
            sp = sigmoid_derivative(z)
            #With this data in hand, we can calculate a new derivative of the cost with respect to the output of the actual hidden layer.
            #The delta of the actual layer is equal to sp * the sum of the products of the delta of the next layer and the weights of the next layer -> matrix product
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            #Once again, with delta in hand the partial derivative of each bias with respect to the input activation is 1 -> the partial derivative is delta
            delta_nabla_b[-l] = delta
            #And in the same way, the partial derivative of the weights with respect to the input activation z=w.x+b, is the sum of the activations of the precedent layer -> matrix product
            delta_nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
        #In the end, each matrix of the gradient has been calculated, we can return the gradient for the training example x,y
        return (delta_nabla_b, delta_nabla_w)
    def quadratic_cost_derivative(self, output_activations, y):
        """This method returns the derivative of the quadratic cost function with respect to the output activations of the last layer"""
        return (output_activations - y)
    def evaluate(self, test_data):
        """This method is used to evaluate the accuracy of the model during its training on a given test data-set"""
        #The "test_results" array contains tuples of (index of the network's highest neuron output, wanted index)
        test_results = [(np.argmax(self.feedforward(x)), y) for x,y in test_data]
        return sum(int(x == y) for x,y in test_results)
    def __repr__(self):
        """We represent a network instance simply by its identifier and its shape"""
        return "Neural network -> " + self.id + " : " + str(self.sizes)
    def plot_accuracy_graph(self, mini_batch_size, eta, total_n_batches, accuracies):
        """This method is executed right after the end of a network training. It plots the training process"""
        plt.plot(total_n_batches, accuracies)
        plt.title("Training of the model \"{0}\" : mini_batch_size = {1}, eta = {2}".format(self.id, mini_batch_size, eta))
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy (measured on the test data)")
        plt.show()


#Activation functions

def sigmoid(x):
    """In this context, we use this method to apply the sigmoid function to the numpy ndarray (n-dimensional array, a sort of "matrix") given in parameter"""
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

