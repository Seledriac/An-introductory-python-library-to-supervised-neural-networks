
# -*- coding:utf-8 -*-

"""Graphic interface module"""

#Third-party Libraries
import numpy as np
import tkinter as tk
import tkinter.font as tkFont
from tkinter import messagebox
from PIL import ImageTk, Image
import pickle
import webbrowser
import os
import sys
sys.path.insert(1, str(os.getcwd()))

#Neural network module
import network


class Interface(tk.Frame):
    """graphic interface class"""


    def __init__(self, fenetre, **kwargs):
        """Displays the main menu"""
        #Fonts
        self.big_font_button = tkFont.Font(family='Calibri', size=20, weight='bold')
        self.medium_font_button = tkFont.Font(family='Calibri', size=14, weight='bold')
        self.font_title = tkFont.Font(family='Calibri', size=36, weight='bold')
        self.main_menu(fenetre, **kwargs)
    

    def main_menu(self, fenetre, **kwargs):
        """Main menu Frame"""
        #Frame creation
        if hasattr(self, 'children'):
            self.destroy()
        tk.Frame.__init__(self, fenetre, width=1180, height=620, bg="#fff2f2", **kwargs)
        self.pack()

        #Github Button
        img_github = ImageTk.PhotoImage(Image.open("hd_recognition_testing/assets/github.jpg").resize((50,50)))
        btn_github = tk.Button(self, image=img_github, command=lambda: webbrowser.open("https://github.com/Seledriac/A-small-pedagogic-python-library-for-supervised-neural-networks/"))
        btn_github.img = img_github
        btn_github.grid(column=0, row=0, padx=50, pady=(0,50))

        #Title
        self.message = tk.Label(self, text="Supervised neural networks\n applied to handwritten digits recognition", bg="#fff2f2", font=self.font_title)
        self.message.grid(column=1, row=0, pady=25)

        #Readme Button
        img_readme = ImageTk.PhotoImage(Image.open("hd_recognition_testing/assets/readme.png").resize((50,50)))
        btn_readme = tk.Button(self, image=img_readme, command=lambda: os.startfile("README.md"))
        btn_readme.img = img_readme
        btn_readme.grid(column=2, row=0, padx=60, pady=(0,50))

        #Button selection frame
        self.btns_frames = tk.LabelFrame(self, padx=50, pady=50, borderwidth=5)
        self.btns_frames.grid(row=1, column=1, columnspan=3, pady=75, padx=(0,180))

        #Menu Buttons
        self.create_model_button = tk.Button(self.btns_frames, text="Create a model", font=self.big_font_button, command=lambda: self.create_model(fenetre, **kwargs))
        self.create_model_button.grid(column=0, row=0, padx=10, pady=10)
        
        self.train_model_button = tk.Button(self.btns_frames, text="Train a model", font=self.big_font_button, command=self.quit)
        self.train_model_button.grid(column = 1, row = 0, padx=10, pady=10)

        self.evaluate_button = tk.Button(self.btns_frames, text="Evaluate", font=self.big_font_button, command=self.quit)
        self.evaluate_button.grid(column = 0, row = 1, padx=10, pady=10)
        
        self.predict_button = tk.Button(self.btns_frames, text="Predict", font=self.big_font_button, command=self.quit)
        self.predict_button.grid(column = 1, row = 1, padx=10, pady=10)


    def create_model(self, fenetre, **kwargs):
        """Model creation Frame"""
        #Frame creation
        self.destroy()
        if hasattr(self, 'hidden_layers_label'):
            delattr(self, 'hidden_layers_label')
        tk.Frame.__init__(self, fenetre, width=1180, height=620, bg="#fff2f2", **kwargs)
        self.pack()

        #Main menu Button
        img_home = ImageTk.PhotoImage(Image.open("hd_recognition_testing/assets/home.png").resize((95,50)))
        btn_home = tk.Button(self, image=img_home, command=lambda: self.main_menu(fenetre, **kwargs))
        btn_home.img = img_home
        btn_home.grid(column=0, row=0, padx=(50,0))
        
        #Title
        self.title = tk.Label(self, text="Model Creation", bg="#fff2f2", font=self.font_title)
        self.title.grid(column=1, row=0, padx=(100,0))

        #Model Validation frame
        self.model_validation_frame = tk.LabelFrame(self, borderwidth=3)
        self.model_validation_frame.grid(row=0, column=2, padx=(0,100), pady=(20,0))
        self.model_validation_label = tk.Label(self.model_validation_frame, text="Model name", font=self.medium_font_button)
        self.model_validation_label.pack()
        self.model_validation_entry = tk.Entry(self.model_validation_frame)
        self.model_validation_entry.pack()
        self.model_validation_button = tk.Button(self.model_validation_frame, text="Create Model", font=self.medium_font_button, command=self.model_validation)
        self.model_validation_button.pack()

        #Model customization frame
        self.custom_frame = tk.LabelFrame(self, padx=50, pady=50, borderwidth=5)
        self.custom_frame.grid(row=1, column=1, columnspan=3, padx=(0,500), pady=(30,0))

        #Input layer Frame
        self.input_layer_frame = tk.LabelFrame(self.custom_frame)
        self.input_layer_frame.grid(row=0, column=0)
        self.input_layer_label = tk.Label(self.input_layer_frame, text="Input Layer", font=self.medium_font_button)
        self.input_layer_label.pack()
        self.input_layer_number = tk.Entry(self.input_layer_frame)
        self.input_layer_number.insert(0,784)
        self.input_layer_number.pack()

        #Hidden layers Frame
        self.hidden_layers = []
        self.hidden_layers_frame = tk.LabelFrame(self.custom_frame)
        self.hidden_layers_frame.grid(row=0, column=1)
        self.add_hidden_layer()
        self.add_hidden_layer()

        #Output layer Frame
        self.output_layer_frame = tk.LabelFrame(self.custom_frame)
        self.output_layer_frame.grid(row=0, column=2, padx=70)
        self.output_layer_label = tk.Label(self.output_layer_frame, text="Output Layer", font=self.medium_font_button)
        self.output_layer_label.pack()
        self.output_layer_number = tk.Entry(self.output_layer_frame)
        self.output_layer_number.insert(0,10)
        self.output_layer_number.pack()

        #Hidden layer adding/deleting buttons
        self.add_hidden_layer_button = tk.Button(self.custom_frame, text="Add a hidden layer", font=self.medium_font_button, command=self.add_hidden_layer)
        self.add_hidden_layer_button.grid(column = 0, row = 1, padx=50, pady=40)
        self.del_hidden_layer_button = tk.Button(self.custom_frame, text="Delete the last hidden layer", font=self.medium_font_button, command=self.del_hidden_layer)
        self.del_hidden_layer_button.grid(column = 1, row = 1, padx=50, pady=40, columnspan=2)
    

    def add_hidden_layer(self):
        """Add a hidden layer in the model creation Frame"""
        if not hasattr(self, 'hidden_layers_label'):
            self.hidden_layers_label = tk.Label(self.hidden_layers_frame, text="Hidden Layer(s)", font=self.medium_font_button)
            self.hidden_layers_label.grid(row=0, column=0, columnspan=10)
        if len(self.hidden_layers) < 5:
            new_hidden_layer = tk.Scale(self.hidden_layers_frame, from_=1, to=128, length=150)
            new_hidden_layer.grid(row=1,column=len(self.hidden_layers), padx=(0,20))
            self.hidden_layers.append(new_hidden_layer)
    

    def del_hidden_layer(self):
        """Delete a hidden layer in the model creation Frame"""
        if len(self.hidden_layers) > 1:
            self.hidden_layers[-1].destroy()
            del self.hidden_layers[-1]
        elif hasattr(self, 'hidden_layers_label'):
            self.hidden_layers[-1].destroy()
            del self.hidden_layers[-1]
            self.hidden_layers_label.destroy()
            delattr(self, 'hidden_layers_label')
    
    def model_validation(self):
        """This method is executed when the model creation validation button is clicked. It creates the model, serlializes it, and shows a recap od the model in a message box to the user"""
        model_name = self.model_validation_entry.get()
        try:
            input_number = int(self.input_layer_number.get())
            output_number = int(self.output_layer_number.get())
        except ValueError:
            messagebox.showerror("Error", "Error, enter a number of neurons for all the layers")
        if model_name and input_number and output_number:
            sizes = [input_number]
            msg = "Model \"{}\" successfully created.\n\nInput layer : {} neurons\n".format(str(self.model_validation_entry.get()), str(input_number))
            for i,layer in enumerate(self.hidden_layers):
                nb_neurons = int(layer.get())
                sizes.append(nb_neurons)
                msg = msg + "Hidden layer {} : {} neurons\n".format(str(i + 1), str(nb_neurons))
            sizes.append(output_number)
            msg = msg + "Output layer : {} neurons".format(str(output_number))
            net = network.Network(model_name, sizes)
            with open("models/hd_recognition/{}.pickle".format(model_name), "wb") as fic:
                pickler = pickle.Pickler(fic)
                pickler.dump(net)
            print(net)
            messagebox.showinfo("Model Info", msg)
        else:
            messagebox.showerror("Error", "Error, missing required fields")
        

#Window creation
fenetre = tk.Tk()
fenetre.geometry("1180x620")
fenetre.title("Neural Networks")
fenetre.configure(bg="#fff2f2")
interface = Interface(fenetre)
interface.mainloop()
interface.destroy()