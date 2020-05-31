
# -*- coding:utf-8 -*-

"""
This is an usage Usage example of the "network" module. In this example, we train a network instance to recognize 28x28 pixels images of handwritten digits
The data used for the training is provided by the mnist database, and loaded by the "mnist_loader" module
"""

#get access to the root of the project
import os
import sys
sys.path.insert(1, str(os.getcwd()))

#The data is loaded with the mnist loader
import mnist_loader

#training, validation, and test data are lists of respectively 50000, 10000, and 10000 tuples. 
#In each tuple, there is the input value x, a column matrix of 28x28 = 784 pixel greyscale values, and the expected output value y, representing the handwritten digit
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
training_data = list(training_data)
validation_data = list(validation_data)
test_data = list(test_data)

#We create a neural network with 28x28 = 784 input neurons, 30 hidden neurons, and 10 output neurons:
# - The activation of the 784 input neurons represent the greyscale value of the 28x28 pixels of a handwritten digit image
# - The hidden neurons add abstraction to the network and hence -> performance
# - The index of the most activated output neuron is the guessed digit 
import network

#We name the model after the other created models
dirs = next(os.walk("models"))[1]
model_count = len(dirs) - 1

#You can tune : 
# - the activation function (sigmoid by default)
# - the regulation of the outputs (none by default)
net = network.Network("hdr_" + str(model_count + 1), [784, 16, 10])
net_description = str(net)
print("\n" + net_description)

#The network is trained with this single line. It calls the SGD training method for the network instance.
#Method call : SGD(training_data, epochs, mini_batch_size, eta, test_data=None, dropout_value = 0.2)
# - training_data is the list of (input,expected_output) tuples (where inputs are 784 column matrixes)
# - epochs is the number of complete training cycles over the training data
# - mini_batch_size is the size of each batch (group of randomly chosen training examples) during the epoch
# - eta (by default 3), is the learning rate, it will be adjusted over epochs
# - min_eta (by default 0.5) is the minimum value the learning will attain while decreasing
# - test_data (None by default) is the test_data over which the network is evaluated after each epoch (for performance tracking, optionnal)
# - verbose (True by default) is wether or not you want to see the progress after each accuracy save (each flag)
# - flags per epoch (5 by default) is how many accuracy flags you want per epoch : at each flag, the learning rate is updated
# - display_weights (True by default) is you want to see the first layer's weights evolving in real time during the training, and save the graphical representation
# - dropout value (0 to 1, None by default), is the proportion of desactivated neurons during each gradient computation
net.SGD(training_data, 5, 10)

#We serialize the trained model as a network object in a file named like itself ("hdr_x")
import pickle
with open("models/hd_recognition/hdr_"+str(model_count + 1)+"", "wb") as saving:
    saver = pickle.Pickler(saving)
    saver.dump(net)

#Performance testing of the network on the validation data
accuracy = str(100 * net.evaluate(validation_data) / 10000)
print("\nTest on the validation data -> Accuracy : {0}%\n".format(accuracy))

#We save the train record
with open("models/hd_recognition/accuracies_ladder.md", "a") as ladder:
    adding = net_description + ", accuracy = " + accuracy + "\n"
    ladder.write(adding)
#And update the accuracies ladder (sorting best accuracies)
with open("models/hd_recognition/accuracies_ladder.md", "r") as ladder:
    content = [net.split("= ") for net in ladder.read().split('\n')]
    content.pop()
    content_updated = sorted([(acc,net) for net,acc in content], reverse = True)
    tostring = "\n".join(["= ".join((net,acc)) for acc,net in content_updated]) + "\n"
with open("models/hd_recognition/accuracies_ladder.md", "w") as ladder:
    ladder.write(tostring)

#Prediction tests

re = False
#The asks variable permits to draw in the same figure each prediction
asks = 0
while re:

    #The user choses a number to predict
    re1 = True
    while re1:
        try:
            chosen_nb = int(input("\nThere is an example for each digit in the custom_test_images folder. Enter the number you want the model to recognize based on theese custom test images : "))
            assert chosen_nb >= 0 and chosen_nb <=9
            re1 = False
        except AssertionError:
            print("\nError, the chosen number isn't a single digit.")
        except ValueError:
            print("\nError, you didn't enter a valid digit.")

    #The image filename is retrieved
    img_filename = "hd_recognition/custom_test_images/test_image_"+str(chosen_nb)+".bmp"

    #Predicting the image
    from PIL import Image
    import numpy as np
    test_image = Image.open(img_filename)
    arr = 1 - np.array(test_image).reshape(784,1) / 255. #Conversion from image to array : 256-RGB to greyscale inverted (1 is black, 0 is white)
    model_activations = net.feedforward(arr)
    print("\nAccording to the AI, the plotted number is {0} !\n".format(np.argmax(model_activations)))

    #Plotting the test_image, and the activations, in subplots (one plots the image, the other plots the model's activation)
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    test_image = mpimg.imread(img_filename)
    if asks == 0:
        fig = plt.figure(figsize = (11, 5))
        plt.show()
        asks = 1
    else:
        plt.clf()
        fig.canvas.draw()
        fig.canvas.flush_events()
    plt.subplot(121)
    plt.title("custom image")
    plt.imshow(test_image)
    plt.subplot(122)
    plt.title("corresponding model activations")
    plt.xlabel("digit")
    plt.ylabel("activation")
    axes = plt.gca()
    axes.set_ylim([0, 1])
    plt.xticks(range(10))
    plt.yticks(np.array(range(11))/10)
    plt.plot(range(10), model_activations)
    #Annotation function to pinpoint the activation on the second subplot
    def annot_max(x, y, ax):
        xmax = x[np.argmax(y)]
        ymax = y.max()
        text = "digit = {}, activation = {:.3f}".format(xmax,ymax)        
        if not ax:
            ax=plt.gca()
        bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
        arrowprops=dict(arrowstyle="->",connectionstyle="angle,angleA=0,angleB=60")
        kw = dict(xycoords='data',textcoords="axes fraction",
                arrowprops=arrowprops, bbox=bbox_props, ha="right", va="top")
        ax.annotate(text, xy=(xmax, ymax), xytext=(xmax/10 - 0.1, ymax - 0.1), **kw)
    annot_max(range(10), model_activations, axes)

    #Ask for a new prediction
    re = str(input("predict another custom digit ? (Y/N) : ")).lower() == "y"


