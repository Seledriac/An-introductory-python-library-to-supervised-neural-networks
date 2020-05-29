
# -*-coding:utf-8 -*

#We will be using the tkinter library
import tkinter

import mnist_loader
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
training_data = list(training_data)
validation_data = list(validation_data)
test_data = list(test_data)
import network
import os

class Interface(tkinter.Frame):
    
    """Notre fenêtre principale.
    Tous les widgets sont stockés comme attributs de cette fenêtre."""
    
    def __init__(self, fenetre, **kwargs):
        tkinter.Frame.__init__(self, fenetre, width=768, height=576, **kwargs)
        self.pack(fill=tkinter.BOTH)
        
        # Création de nos widgets
        self.message = tkinter.Label(self, text="Neural Networks applied to handwritten digits recognition")
        self.message.pack()
        
        self.bouton_quitter = tkinter.Button(self, text="Quitter", command=self.quit)
        self.bouton_quitter.pack(side="left")
        
        #model creation button
        self.bouton_create_model = tkinter.Button(self, text="Create a model", fg="red", command=self.create_model)
        self.bouton_create_model.pack(side = "right")

        #model training button
        self.bouton_train_model = tkinter.Button(self, text="Train the model", fg="red", command=self.train_model)
        self.bouton_train_model.pack()

        #model prediction button
        self.bouton_predict = tkinter.Button(self, text="Predict", fg="red", command=self.predict)
        self.bouton_predict.pack()

    def create_model(self):
        dirs= next(os.walk("trainings"))[1]
        model_count = len(dirs)
        #We can of course choose any activation function, by default it will be sigmoid
        global net
        net = network.Network("hdr_" + str(model_count + 1), [784, 36, 10])
    
    def train_model(self):
        net.SGD(training_data, 20, 10, 3, test_data = test_data)
    
    def predict(self):
        re = True
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
            img_filename = "custom_test_images/test_image_"+str(chosen_nb)+".bmp"

            #Predicting the image
            from PIL import Image
            import numpy as np
            test_image = Image.open(img_filename)
            arr = 1 - np.array(test_image).reshape(784,1) / 255. #Conversion from image to array : 256-RGB to greyscale inverted (1 is black, 0 is white)
            model_activations = net.feedforward(arr)
            print("\nAccording to the IA, the plotted number is {0} !\n".format(np.argmax(model_activations)))

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

fenetre = tkinter.Tk()
interface = Interface(fenetre)

interface.mainloop()
interface.destroy()
