# Supervised Neural Networks

* This repository contains a network module, with a class which models a neural Network you can train in a supervised learning paradigm with the standard SGD (Stochastic Gradient Descent)/backpropagation algorithm method

* How to setup your testing environment :
    - see "practical_commands.md" 

* Requirements :
    - python 3 (includes Tkinter)
    - numpy
    - matplotlib
    - Pillow
    - ImageMagick

* How to use the network module :
    - Import it : "import network"
    - Create a network instance, specifying its identifier, its shape with a list of numbers, the activation function, and the output regulation function :
    	- The first and last number of the list will always respectively describe the input and output layers of the created network.
    	- There are 3 possible activation functions : sigmoid, relu, and tanh
    - Train your network, tuning the hyper-parameters, using "net.SGD(training_data, epochs, mini_batch_size, learning_rate, min_eta, test_data, verbose, flags_per_epoch, display_weights, dropout_value)"
    	- The training/test/validation data must be lists of tuples of a numpy vector x and a digit y : [(x1 , y1), ... ,(xn , yn)] (where n is the training/validation data-set's size), where x is are numpy vectors, representing the inputs given to the network, and y are the corresponding expected outputs
    - Save your trained model as a serialized Network object in a file
    - Track the performances of your models during and after training, end up with the optimal configuration to solve your problem, and try to predict with the model on custom examples
    - You can use your own training/testing/validation data sets and extraction scripts (in the "mnist_loader.py" style) for them to implement the networks in any AI problem.

* Usage examples :
    - handwritten digits recognition : we use the mnist dataset : 50000 training images, 10000 test images, and 10000 validation images

* Side notes :
    - the data used for the hdr (handwritten digits recognition) models is loaded from the mnist database by the "mnist_loader" library
    - the custom paint test images are 256-RGB format .bmp files
    - WARNING : the performance on the mnist dataset =/= the performance on custom handwritten digits (for overfitting reasons)
    - WARNING : the only currently correctly working combination for regu/activation function is sigmoid with no regulation : about output activations : relu doesn't work without softmax and tanh is very unstable and regarding regulation functions, softmax and normalization have exploding or vanishing output values problems. Overall, with regulation functions, the results are very bad.
    - The "test_hd.py" script trains a model, stores it in the "models/hd_recognition" folder, and the tracked training process in "trainings". It can also live display the weights training, and once the training is done, let the user use the model to predict on custom examples.
    - Use the "gui.py" graphical interface to get a simplified usage of the network creation, model training using the mnist dataset, and user-friendly image recognition predictions
    - The models are saved in the "models" folder, and the trainings in the "trainings" folder
    - After training, the kept model state is the one which had the highest accuracy -> caution : sometimes, there is a bug on the line which saves the best network state : 'saved_state = sorted(states)[-1]' in the network module. It randomly happens for some trainings, and more when output values are extreme.


This library was inspired by the Michael Nielsen's e-book : Neural Networks and deep learning (http://neuralnetworksanddeeplearning.com).
"mnist_loader.py" script written by Michael Nielsen
    
	
