
# -*-coding:utf-8 -*

import numpy as np
import random
from matplotlib import pyplot as plt

class Network():
    def __init__(self, id, sizes):
        self.id = str(id)
        self.sizes = sizes
        self.num_layers = len(sizes)
        self.biases = [np.random.randn(y,1) for y in sizes[1:]]
        self.weights = [np.random.randn(y,x) for x,y in zip(sizes[:-1],sizes[1:])]
    def feedforward(self, x):
        for w,b in zip(self.weights, self.biases):
            x = sigmoid(np.dot(w, x) + b) 
        return x
    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        training_data = list(training_data)
        n = len(training_data)
        if test_data:
            test_data = list(test_data)
            n_test = len(test_data)
            print("\nBeginning of the standard SGD method. Accuracy of the model at this state (with random weights and biases) : {}%\n".format(100 * self.evaluate(test_data) / n_test))
            print("The network will be trained with :\n- {0} epochs,\n- a mini-batch size of {1},\n- a learning rate of eta = {2}\n".format(epochs, mini_batch_size, eta))
            accuracies = [100 * self.evaluate(test_data) / n_test]
        for i in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0,n,mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                total_n_batches += 1
                accuracy = 100 * self.evaluate(test_data) / n_test
                accuracies.append(accuracy)
                print("\nEpoch n°{0} completed. Accuracy of the model at this state : {1}%".format(i + 1, accuracy))
            else:
                print("\nEpoch n°{0} completed.".format(i))
        if test_data:
            self.plot_accuracy_graph(mini_batch_size, eta, range(epochs), accuracies)
    def update_mini_batch(self, mini_batch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x,y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x,y)
            nabla_b = [nb + dnb for nb,dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw,dnw in zip(nabla_w, delta_nabla_w)]
        self.biases = [b - eta * (nb / len(mini_batch)) for b,nb in zip(self.biases, nabla_b)]
        self.weights = [w - eta * (nw / len(mini_batch)) for w,nw in zip(self.weights, nabla_w)]
    def backprop(self, x, y):
        delta_nabla_b = [np.zeros(b.shape) for b in self.biases]
        delta_nabla_w = [np.zeros(w.shape) for w in self.weights]
        activation = x
        activations = [x]
        zs = []
        for b,w in zip(self.biases, self.weights):
            z = np.dot(w,activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        delta = self.quadratic_cost_derivative(activations[-1], y) * sigmoid_derivative(zs[-1])
        delta_nabla_b[-1] = delta
        delta_nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_derivative(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            delta_nabla_b[-l] = delta
            delta_nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (delta_nabla_b, delta_nabla_w)
    def quadratic_cost_derivative(self, output_activations, y):
        return (output_activations - y)
    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)),y) for x,y in test_data]
        return sum(int(x == y) for x,y in test_results)
    def __repr__(self):
        return "Neural network -> " + self.id + " : " + str(self.sizes)
    def plot_accuracy_graph(self, mini_batch_size, eta, total_n_batches, accuracies):
        plt.plot(total_n_batches, accuracies)
        plt.title("Training of the model \"{0}\" : mini_batch_size = {1}, eta = {2}".format(self.id, mini_batch_size, eta))
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy (measured on the test data)")
        plt.show()
    

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))



