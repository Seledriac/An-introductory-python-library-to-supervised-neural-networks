
# -*-coding:utf-8 -*

import numpy as np
import random
from matplotlib import pyplot as plt
plt.ion()
from PIL import Image
import matplotlib.image as mpimg
from math import sqrt, ceil

class Network():
    def __init__(self, id, sizes, activation_function_name = 'sigmoid'):
        self.id = str(id)
        self.sizes = sizes
        self.num_layers = len(sizes)
        self.biases = [np.random.randn(y,1) for y in sizes[1:]]
        self.weights = [np.random.randn(y,x) for x,y in zip(sizes[:-1],sizes[1:])]
        self.activation_function_name = activation_function_name
    def feedforward(self, x):
        for w,b in zip(self.weights, self.biases):
            x = activation_function(np.dot(w, x) + b, self.activation_function_name) 
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
            import os
            dirs= next(os.walk("trainings"))[1]
            model_count = len(dirs) + 1
            file_count = 0
            os.mkdir("trainings/training_{}".format(str(model_count)))
            os.mkdir("trainings/training_{}/weight_plots".format(str(model_count)))
            fig_size = ceil(sqrt(len(self.weights[0])))
            fig = plt.figure(figsize = (fig_size, fig_size))
            fig.suptitle("Weights live training (first hidden layer)", fontsize=16)
            self.update_plot_weights(fig, fig_size, model_count, file_count)
            plt.show()
        for i in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0,n,mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                file_count += 1
                self.update_plot_weights(fig, fig_size, model_count, file_count)
                accuracy = 100 * self.evaluate(test_data) / n_test
                accuracies.append(accuracy)
                print("\nEpoch n°{0} completed. Accuracy of the model at this state : {1}%".format(i + 1, accuracy))
            else:
                print("\nEpoch n°{0} completed.".format(i))
        if test_data:
            import glob
            os.chdir("trainings/training_{}/weight_plots".format(str(model_count)))
            gif_name = 'training_animation'
            file_list = glob.glob('*.png') 
            list.sort(file_list, key=lambda x: int(x.split('_')[1].split('.png')[0])) 
            with open('image_list.txt', 'w') as file:
                for item in file_list:
                    file.write("%s\n" % item)
            os.system('convert @image_list.txt {}.gif'.format(gif_name))
            os.remove('image_list.txt')
            os.chdir("../../../") 
            self.plot_accuracy_graph(mini_batch_size, eta, range(epochs + 1), accuracies, model_count)
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
            activation = activation_function(z, self.activation_function_name)
            activations.append(activation)
        delta = self.quadratic_cost_derivative(activations[-1], y) * activation_function_derivative(zs[-1], self.activation_function_name)
        delta_nabla_b[-1] = delta
        delta_nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = activation_function_derivative(z, self.activation_function_name)
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
    def update_plot_weights(self, fig, fig_size, model_count, file_count):
        for j,weights in enumerate(self.weights[0]):
            fig.add_subplot(fig_size, fig_size, j + 1)
            plt.gca().axes.xaxis.set_ticklabels([])
            plt.gca().axes.yaxis.set_ticklabels([])
            plt.gca().axes.get_xaxis().set_visible(False)
            plt.gca().axes.get_yaxis().set_visible(False)
            plt.imshow(weights.reshape(28,28))
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.savefig("trainings/training_{}/weight_plots/epoch_{}".format(str(model_count),str(file_count)))
    def plot_accuracy_graph(self, mini_batch_size, eta, epochs, accuracies, model_count):
        plt.figure(figsize = (8, 8))
        plt.plot(epochs, accuracies)
        plt.title("Training of the model \"{0}\" : mini_batch_size = {1}, eta = {2}".format(self.id, mini_batch_size, eta))
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy (measured on the test data)")
        plt.savefig("trainings/training_{}/training_graph".format(str(model_count)))
            
    

def activation_function(x, activation_function_name):
    if activation_function_name == 'sigmoid':
        return 1 / (1 + np.exp(-x))
    elif activation_function_name == 'relu':
        return np.maximum(0,x)
    elif activation_function_name == 'tanh':
        return np.tanh(x)


def activation_function_derivative(x, activation_function_name):
    if activation_function_name == 'sigmoid':
        return activation_function(x, activation_function_name) * (1 - activation_function(x, activation_function_name))
    elif activation_function_name == 'relu':
        derivative = []
        for activation in x:
            if activation <= 0:
                derivative.append(0)
            else:
                derivative.append(1)
        return np.array(derivative).reshape(x.shape)
    elif activation_function_name == 'tanh':
        return 1 - activation_function(x, activation_function_name) * activation_function(x, activation_function_name)
    



