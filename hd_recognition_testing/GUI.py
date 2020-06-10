
# -*- coding:utf-8 -*-

"""Handwritten digits recognition Graphic interface module : training done with the mnist dataset"""

# Third-party Libraries
import numpy as np
import tkinter as tk
import tkinter.font as tkFont
from tkinter import messagebox
from tkinter import filedialog
from PIL import ImageTk, Image
import pickle
import webbrowser
import os
import sys
sys.path.insert(1, str(os.getcwd()))

# Neural network module
import network



class Interface(tk.Frame):
    """graphic interface class"""


# ------------------------------------------------------------------------------__init__------------------------------------------------------------------------------------------------

    def __init__(self, fenetre, **kwargs):
        """Displays the main menu"""
        # Fonts
        self.big_font_button = tkFont.Font(family='Calibri', size=20, weight='bold')
        self.medium_large_font_button = tkFont.Font(family='Calibri', size=16, weight='bold')
        self.medium_font_button = tkFont.Font(family='Calibri', size=14, weight='bold')
        self.font_title = tkFont.Font(family='Calibri', size=36, weight='bold')

        # Display main menu
        self.main_menu(fenetre, **kwargs)


# ------------------------------------------------------------------------------Main Menu Interface--------------------------------------------------------------------------------------

    def main_menu(self, fenetre, **kwargs):
        """Main menu Frame"""
        # Frame creation
        if hasattr(self, 'children'):
            self.destroy()
        tk.Frame.__init__(self, fenetre, width=1180, height=620, bg="#fff2f2", **kwargs)
        self.pack()

        # Github Button
        img_github = ImageTk.PhotoImage(Image.open("hd_recognition_testing/assets/github.jpg").resize((50,50)))
        btn_github = tk.Button(self, image=img_github, command=lambda: webbrowser.open("https://github.com/Seledriac/A-small-pedagogic-python-library-for-supervised-neural-networks/"))
        btn_github.img = img_github
        btn_github.grid(column=0, row=0, padx=50, pady=(0,50))

        # Title
        title = tk.Label(self, text="Supervised neural networks\n applied to handwritten digits recognition", bg="#fff2f2", font=self.font_title)
        title.grid(column=1, row=0, pady=25)

        # Readme Button
        img_readme = ImageTk.PhotoImage(Image.open("hd_recognition_testing/assets/readme.png").resize((50,50)))
        btn_readme = tk.Button(self, image=img_readme, command=lambda: os.startfile("README.md"))
        btn_readme.img = img_readme
        btn_readme.grid(column=2, row=0, padx=60, pady=(0,50))

        # Button selection frame
        btns_frames = tk.LabelFrame(self, padx=50, pady=50, borderwidth=5)
        btns_frames.grid(row=1, column=1, columnspan=3, pady=(65,80), padx=(0,180))

        # Menu Buttons
        create_model_button = tk.Button(btns_frames, text="Create a model", font=self.big_font_button, command=lambda: self.create_model(fenetre, **kwargs))
        create_model_button.grid(column=0, row=0, padx=10, pady=10)
        
        train_model_button = tk.Button(btns_frames, text="Train a model", font=self.big_font_button, command=lambda: self.train_model(fenetre, **kwargs))
        train_model_button.grid(column = 1, row = 0, padx=10, pady=10)

        evaluate_button = tk.Button(btns_frames, text="Accuracy Ladder", font=self.big_font_button, command=lambda: self.models_ladder(fenetre, **kwargs))
        evaluate_button.grid(column = 0, row = 1, padx=10, pady=10)
        
        predict_button = tk.Button(btns_frames, text="Predict", font=self.big_font_button, command=lambda: self.choose_prediction(fenetre, **kwargs))
        predict_button.grid(column = 1, row = 1, padx=10, pady=10)


# ------------------------------------------------------------------------------Model Creation Interface------------------------------------------------------------------------------------

    def create_model(self, fenetre, **kwargs):
        """Model creation Frame"""
        # Frame creation
        self.destroy()
        if hasattr(self, 'hidden_layers_label'):
            delattr(self, 'hidden_layers_label')
        tk.Frame.__init__(self, fenetre, width=1180, height=620, bg="#fff2f2", **kwargs)
        self.pack()

        # Main menu Button
        img_home = ImageTk.PhotoImage(Image.open("hd_recognition_testing/assets/home.png").resize((95,50)))
        btn_home = tk.Button(self, image=img_home, command=lambda: self.main_menu(fenetre, **kwargs))
        btn_home.img = img_home
        btn_home.grid(column=0, row=0)
        
        # Title
        title = tk.Label(self, text="Model Creation", bg="#fff2f2", font=self.font_title)
        title.grid(column=1, row=0)

        # Model Validation frame
        model_creation_validation_frame = tk.LabelFrame(self, borderwidth=3)
        model_creation_validation_frame.grid(row=0, column=2, pady=(20,0))
        model_creation_validation_label = tk.Label(model_creation_validation_frame, text="Model name", font=self.medium_font_button)
        model_creation_validation_label.pack()
        self.model_creation_validation_entry = tk.Entry(model_creation_validation_frame)
        self.model_creation_validation_entry.pack()
        model_creation_validation_button = tk.Button(model_creation_validation_frame, text="Create Model", font=self.medium_font_button, command=self.model_creation_validation)
        model_creation_validation_button.pack()

        # Model customization frame
        creation_custom_frame = tk.LabelFrame(self, padx=50, pady=50, borderwidth=5)
        creation_custom_frame.grid(row=1, column=0, columnspan=3, pady=(30,0))

        # Input layer Frame
        input_layer_frame = tk.LabelFrame(creation_custom_frame)
        input_layer_frame.grid(row=0, column=0)
        input_layer_label = tk.Label(input_layer_frame, text="Input Layer", font=self.medium_font_button)
        input_layer_label.pack()
        self.input_layer_number = tk.Entry(input_layer_frame)
        self.input_layer_number.insert(0,784)
        self.input_layer_number.pack()

        # Hidden layers Frame
        self.hidden_layers = []
        self.hidden_layers_frame = tk.LabelFrame(creation_custom_frame)
        self.hidden_layers_frame.grid(row=0, column=1)
        self.add_hidden_layer()
        self.add_hidden_layer()

        # Output layer Frame
        output_layer_frame = tk.LabelFrame(creation_custom_frame)
        output_layer_frame.grid(row=0, column=2, padx=70)
        output_layer_label = tk.Label(output_layer_frame, text="Output Layer", font=self.medium_font_button)
        output_layer_label.pack()
        self.output_layer_number = tk.Entry(output_layer_frame)
        self.output_layer_number.insert(0,10)
        self.output_layer_number.pack()

        # Hidden layer adding/deleting buttons
        add_hidden_layer_button = tk.Button(creation_custom_frame, text="Add a hidden layer", font=self.medium_font_button, command=self.add_hidden_layer)
        add_hidden_layer_button.grid(column = 0, row = 1, padx=50, pady=40)
        del_hidden_layer_button = tk.Button(creation_custom_frame, text="Delete the last hidden layer", font=self.medium_font_button, command=self.del_hidden_layer)
        del_hidden_layer_button.grid(column = 1, row = 1, padx=50, pady=40, columnspan=2)    

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
    
    def model_creation_validation(self):
        """This method is executed when the model creation validation button is clicked. It creates the model, serlializes it, and shows a recap od the model in a message box to the user"""
        model_name = self.model_creation_validation_entry.get()
        try:
            input_number = int(self.input_layer_number.get())
            output_number = int(self.output_layer_number.get())
        except ValueError:
            messagebox.showerror("Error", "Error : enter a number of neurons for all the layers")
        if model_name and input_number and output_number:
            sizes = [input_number]
            msg = "Model \"{}\" successfully created.\n\nInput layer : {} neurons\n".format(str(self.model_creation_validation_entry.get()), str(input_number))
            for i,layer in enumerate(self.hidden_layers):
                nb_neurons = int(layer.get())
                sizes.append(nb_neurons)
                msg = msg + "Hidden layer {} : {} neurons\n".format(str(i + 1), str(nb_neurons))
            sizes.append(output_number)
            msg = msg + "Output layer : {} neurons\n\nActivation function : sigmoid (by default)".format(str(output_number))
            net = network.Network(model_name, sizes)
            with open("models/hd_recognition/{}.pickle".format(model_name), "wb") as fic:
                pickler = pickle.Pickler(fic)
                pickler.dump(net)
            messagebox.showinfo("Model Info", msg)
        else:
            messagebox.showerror("Error", "Error : missing required fields")


# ------------------------------------------------------------------------------Model Training Interface------------------------------------------------------------------------------------

    def train_model(self, fenetre, **kwargs):
        """Model training specs Frame"""
        # Frame creation
        self.destroy()
        tk.Frame.__init__(self, fenetre, width=1180, height=620, bg="#fff2f2", **kwargs)
        self.pack()

        re = True
        while re:
            try:
                # Model file opening prompt
                self.model_filename = filedialog.askopenfilename(initialdir="models/hd_recognition", title="Choose the model", filetypes=(("pickle files","*.pickle"), ("model files","*.model"), ("all files", "*.*")))
                assert self.model_filename
                re = False
            except:
                messagebox.showerror("Error", "Error : please select a model file")

        # Main menu Button
        img_home = ImageTk.PhotoImage(Image.open("hd_recognition_testing/assets/home.png").resize((95,50)))
        btn_home = tk.Button(self, image=img_home, command=lambda: self.main_menu(fenetre, **kwargs))
        btn_home.img = img_home
        btn_home.grid(column=0, row=0, padx=(25,0))
        
        # Title
        title = tk.Label(self, text="Model Training\n(mnist dataset)", bg="#fff2f2", font=self.font_title)
        title.grid(column=1, row=0, pady=80, padx=(200,0))

        # Model training validation frame
        model_training_validation_frame = tk.LabelFrame(self, borderwidth=3)
        model_training_validation_frame.grid(row=0, column=2, padx=(200,0), pady=(10,0))
        model_training_validation_button = tk.Button(model_training_validation_frame, text="Train", font=self.medium_large_font_button, command=lambda: self.model_training(fenetre, **kwargs))
        model_training_validation_button.pack()

        # Model training customization frame
        training_custom_frame = tk.LabelFrame(self, padx=50, pady=50, borderwidth=5)
        training_custom_frame.grid(row=1, column=0, columnspan=100, padx=(0,15))

        # Epochs Frame
        epochs_frame = tk.LabelFrame(training_custom_frame)
        epochs_frame.grid(row=0, column=0)
        epochs_label = tk.Label(epochs_frame, text="Epochs", font=self.medium_font_button)
        epochs_label.pack()
        self.epochs_number = tk.Entry(epochs_frame)
        self.epochs_number.insert(0,3)
        self.epochs_number.pack()

        # Batch size Frame
        batch_size_frame = tk.LabelFrame(training_custom_frame)
        batch_size_frame.grid(row=0, column=2, padx=70)
        batch_size_label = tk.Label(batch_size_frame, text="batch size", font=self.medium_font_button)
        batch_size_label.pack()
        self.batch_size_number = tk.Entry(batch_size_frame)
        self.batch_size_number.insert(0,10)
        self.batch_size_number.pack()

        # Display weights checkbox
        display_weights_frame = tk.LabelFrame(training_custom_frame)
        display_weights_frame.grid(row=0, column=3)
        self.display_weights_value = tk.IntVar()
        display_weights_cb = tk.Checkbutton(display_weights_frame, text="Dynamically display the weights of the first layer", font=self.medium_font_button, variable=self.display_weights_value)
        display_weights_cb.pack()

    def model_training(self, fenetre, **kwargs):
        """Model training Frame"""

        # Training values retrieving
        disp_weights = bool(self.display_weights_value.get())
        try:
            epochs = int(self.epochs_number.get())
            batch_size = int(self.batch_size_number.get())
        except ValueError:
            messagebox.showerror("Error", "Error : please enter a numeric value for each field")
        
        if epochs and batch_size:
            # Frame creation
            self.destroy()
            tk.Frame.__init__(self, fenetre, width=1180, height=620, bg="#fff2f2", **kwargs)

            # Main menu Button
            img_home = ImageTk.PhotoImage(Image.open("hd_recognition_testing/assets/home.png").resize((95,50)))
            btn_home = tk.Button(self, image=img_home, command=lambda: self.main_menu(fenetre, **kwargs))
            btn_home.img = img_home
            btn_home.grid(column=0, row=0)

            # Training trigger button
            doIt = tk.Button(self, text="Start the Training", command=lambda: self.start_training(epochs, batch_size, disp_weights), font=self.big_font_button)
            doIt.grid(row=0, column=1, pady=20)

            # Training logs textbox
            textbox_frame = tk.LabelFrame(self)
            textbox_frame.grid(row=1, column=0, columnspan=2)
            self.output = tk.Text(textbox_frame, width=110, height=30, bg='black', fg='white')
            self.output.pack(side=tk.LEFT)

            # Scrollbar
            scrollbar = tk.Scrollbar(textbox_frame, orient="vertical", command = self.output.yview)
            scrollbar.pack(side=tk.RIGHT, fill="y")
            self.output['yscrollcommand'] = scrollbar.set

            self.pack()
        else:
            messagebox.showerror("Error", "Error : missing required fields")

    def start_training(self, epochs, batch_size, disp_weights):
        """This method executes the SGD training method on a given model"""
        # Importing the mnist dataset
        import mnist_loader
        training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
        training_data = list(training_data)
        validation_data = list(validation_data)
        test_data = list(test_data)

        # Model training via SGD
        with open(self.model_filename, "rb") as fic:
            unpickler = pickle.Unpickler(fic)
            net = unpickler.load()
        self.output.insert(tk.END, "\n" + str(net) + "\n")
        self.update_idletasks()
        net.SGD(training_data, epochs, batch_size, test_data=test_data, display_weights=disp_weights, gui=self)

        # Model saving
        with open("models/hd_recognition/{}.pickle".format(net.id), "wb") as saving:
            saver = pickle.Pickler(saving)
            saver.dump(net)

        # Performance test of the network on the validation data
        accuracy = str(100 * net.evaluate(validation_data) / 10000)
        self.output.insert(tk.END, "\nTest on the validation data -> Accuracy : {0}%\n".format(accuracy))
        self.update_idletasks()
        self.output.see("end")

        # Ladder update
        with open("models/hd_recognition/accuracies_ladder.md", "a") as ladder:
            adding = str(net) + " --> accuracy = " + accuracy + "\n"
            ladder.write(adding)
        with open("models/hd_recognition/accuracies_ladder.md", "r") as ladder:
            shove_percent = ladder.read().replace("%", "")
            content = [net.split("= ") for net in shove_percent.split('\n')]
            content.pop()
            content_updated = sorted([(acc,net) for net,acc in content], reverse = True)
            tostring = "%\n".join(["= ".join((net,acc)) for acc,net in content_updated]) + "%\n"
        with open("models/hd_recognition/accuracies_ladder.md", "w") as ladder:
            ladder.write(tostring)


# ------------------------------------------------------------------------------Models Ladder Interface------------------------------------------------------------------------------------

    def models_ladder(self, fenetre, **kwargs):
        """Models ladder frame"""
        # Frame creation
        self.destroy()
        tk.Frame.__init__(self, fenetre, width=1180, height=620, bg="#fff2f2", **kwargs)

        # Main menu Button
        img_home = ImageTk.PhotoImage(Image.open("hd_recognition_testing/assets/home.png").resize((95,50)))
        btn_home = tk.Button(self, image=img_home, command=lambda: self.main_menu(fenetre, **kwargs))
        btn_home.img = img_home
        btn_home.grid(column=0, row=0)

        # Ladder label
        ladder_label = tk.Label(self, text="Models Accuracy Ladder", font=self.font_title, bg="#fff2f2")
        ladder_label.grid(row=0, column=1, padx=(0,150), pady=20)

        # Ladder textbox
        textbox_frame = tk.LabelFrame(self)
        textbox_frame.grid(row=1, column=0, columnspan=2)
        output = tk.Text(textbox_frame, width=100, height=20, font=self.medium_font_button)
        output.pack(side=tk.LEFT)
        with open("models/hd_recognition/accuracies_ladder.md", "r") as ladder:
            content = ladder.read()
        output.insert(tk.END, content)
        self.update_idletasks()
        output.see("end")

        # Scrollbar
        scrollbar = tk.Scrollbar(textbox_frame, orient="vertical", command = output.yview)
        scrollbar.pack(side=tk.RIGHT, fill="y")
        output['yscrollcommand'] = scrollbar.set

        self.pack()


# ------------------------------------------------------------------------------Prediction Interface---------------------------------------------------------------------------------------

    def choose_prediction(self, fenetre, **kwargs):
        """Prediction style choice frame"""
        # Frame creation
        self.destroy()
        tk.Frame.__init__(self, fenetre, width=1180, height=620, bg="#fff2f2", **kwargs)
        self.pack()

        # Main menu Button
        img_home = ImageTk.PhotoImage(Image.open("hd_recognition_testing/assets/home.png").resize((95,50)))
        btn_home = tk.Button(self, image=img_home, command=lambda: self.main_menu(fenetre, **kwargs))
        btn_home.img = img_home
        btn_home.grid(column=0, row=0, columnspan=2, padx=(0,1000), pady=(15,0))

        # Choice buttons
        choice_custom = tk.Button(self, text="Predict with custom test images", font=self.big_font_button, command=self.custom_prediction_frame(fenetre, **kwargs))
        choice_custom.grid(row=1, column=0, padx=(100,0), pady=(200))
        choice_live = tk.Button(self, text="Live prediction", font=self.big_font_button, command=self.live_prediction_frame(fenetre, **kwargs))
        choice_live.grid(row=1, column=1, padx=(0,200), pady=(200))
    
    def custom_prediction_frame(self, fenetre, **kwargs):
        """Custom images prediction frame"""
        pass

    def live_prediction_frame(self, fenetre, **kwargs):
        """Live predictions of the numbers drew by the user"""
        pass



# Window creation
fenetre = tk.Tk()
fenetre.geometry("1180x620")
fenetre.title("Neural Networks")
fenetre.configure(bg="#fff2f2")
interface = Interface(fenetre)
interface.mainloop()

