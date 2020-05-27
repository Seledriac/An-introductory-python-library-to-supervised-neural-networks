
# -*-coding:utf-8 -*

"""
This is an usage Usage example of the "network" module. In this example, we train a network instance to recognize 28x28 pixels images of handwritten digits
The data used for the training is provided by the mnist database, and loaded by the "mnist_loader" module
"""

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
# import network
# net = network.Network("hdr_2", [784, 30, 30, 10])
# print("\n",net, sep="")

#The network is trained with this single line. It calls the SGD training method for the network instance.
#Method call : SGD(training_data, epochs, mini_batch_size, eta, test_data=None)
# - training_data is the list of (input,expected_output) tuples (where inputs are 784 column matrixes)
# - epochs is the number of complete training cycles over the training data
# - mini_batch_size is the size of each batch (group of randomly chosen training examples) during the epoch
# - eta is the learning rate
# - test_data is the test_data over which the network is evaluated after each epoch (for performance tracking, optionnal)
# net.SGD(training_data, 60, 10, 3.0, test_data = test_data)

# We serialize the trained model as a network object in the "hdr" (handwritten_digits_recognizer) file
import pickle
# with open("hdr_2", "wb") as saving:
#     saver = pickle.Pickler(saving)
#     saver.dump(net)

#Deserialization of a saved model as a network object
with open("hdr_1", "rb") as retrieving_model:
    retriever = pickle.Unpickler(retrieving_model)
    dn = retriever.load()

#Performance testing of the deserialized network 
print("\nDeserialized network -> \"{0}\"".format(dn))
accuracy = 100 * dn.evaluate(validation_data) / 10000
print("\nTest on the validation data -> deserialized network accuracy : {0}%\n".format(accuracy))
print("Wow ! Your model can recognize 28x28px images of handwritten digits with an accuracy of {0}% !\n".format(accuracy))


#Prediction tests

re = True
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
    # img_filename = "custom_test_images/mnist_example.bmp"
    img_filename = "custom_test_images/test_image_"+str(chosen_nb)+".bmp"

    #Predicting the image
    from PIL import Image
    import numpy as np
    test_image = Image.open(img_filename)
    arr = 1 - np.array(test_image).reshape(784,1) / 255. #Conversion from image to array : 256-RGB to greyscale inverted (1 is black, 0 is white)
    model_activations = dn.feedforward(arr)
    print("\nAccording to the IA, the plotted number is {0} !\n".format(np.argmax(model_activations)))

    #Plotting the test_image, and the activations, in subplots (one plots the image, the other plots the model's activation)
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    test_image = mpimg.imread(img_filename)
    plt.figure(figsize = (11, 5))
    plt.title("Prediction results")
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
    def annot_max(x, y, ax=axes):
        xmax = x[np.argmax(y)]
        ymax = y.max()
        text= "digit = {}, activation = {:.3f}".format(xmax,ymax)
        if not ax:
            ax=plt.gca()
        bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
        arrowprops=dict(arrowstyle="->",connectionstyle="angle,angleA=0,angleB=60")
        kw = dict(xycoords='data',textcoords="axes fraction",
                arrowprops=arrowprops, bbox=bbox_props, ha="right", va="top")
        ax.annotate(text, xy=(xmax, ymax), xytext=(xmax/10 - 0.1, ymax - 0.1), **kw)
    annot_max(range(10), model_activations)
    plt.show()

    #Ask for a new prediction
    re = str(input("predict another custom digit ? (O/N) : ")).lower() == "o"


