
* This repository contains a network module, with a class which models a neural Network you can train in a supervised learning paradigm with the standard SGD (Stochastic Gradient Descent)/backpropagation algorithm method

* How to setup your testing environment :
    - see "practical_commands.txt" 

* How to use the network module :
    - Import it : "import network"
    - Create a network instance, specifying its identifier, and its shape with a list of numbers :
    	- The first and last number of the list will always respectively describe the input and output layers of the created network.
    - Train your network, specifying your hyper-parameters, using "net.SGD(training_data, epochs, mini_batch_size, learning_rate, test_data)"
    	- The training/test/validation data must be lists of tuples of a numpy vector x and a digit y : [(x1 , y1), ... ,(xn , yn)] (where n is the training/validation data-set's size)
    	- The numpy vector x is the input given to the network, and y is this expected output
    - Save your trained model as a serialized Network object in a file
    - Track the performances of your model during training, and after training, try to predict with the model on custom examples
    - Use your model to solve a problem :)

* Usage examples :
    - test.py
    - test_hd.py (hd for handwritten_digits)

* Side notes :
    - the hdr_1 is a 1 hidden layer neural network, with learning parameters : 30, 10, 3.0 
    - the hdr_2 is a 2 hidden layers neural network, with learning parameters : 60, 10, 3.0
    - the hdr_3 is a 3 hidden layers neural network, with learning parameters : 500, 500, 10
    - the training graphs for hdr_1/2/3 are represented in the hd_training_example_1/2/3.png images
    - WARNING : the performance on the mnist dataset =/= the performance on custom handwritten digits
    - During the training time, the hdr_3 spent the last 350 epochs stuck at 85% accuracy, the learning rate was 10.0

*This library was inspired by the Michael Nielsen's e-book : Neural Networks and deep learning 
*http://neuralnetworksanddeeplearning.com/
    
	
