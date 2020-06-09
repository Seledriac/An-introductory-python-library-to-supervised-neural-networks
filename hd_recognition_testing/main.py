
# -*- coding:utf-8 -*-

"""Graphic interface module"""

import numpy as np
import tkinter
import tkinter.font as tkFont
from PIL import ImageTk, Image
import webbrowser
import os

class Interface(tkinter.Frame):
    
    """graphic interface class"""

    def __init__(self, fenetre, **kwargs):

        #Frame creation, font creations
        tkinter.Frame.__init__(self, fenetre, width=1280, height=720, **kwargs)
        self.pack()
        font_button = tkFont.Font(family='Calibri', size=20, weight='bold')
        font_title = tkFont.Font(family='Calibri', size=36, weight='bold')

        #Github Button
        img_github = ImageTk.PhotoImage(Image.open("hd_recognition_testing/assets/github.jpg").resize((50,50)))
        btn_github = tkinter.Button(self, image=img_github, command=lambda: webbrowser.open("https://github.com/Seledriac/A-small-pedagogic-python-library-for-supervised-neural-networks/"))
        btn_github.img = img_github
        btn_github.grid(column=0, row=0, padx=50)

        #Title
        self.message = tkinter.Label(self, text="Supervised neural networks\n applied to handwritten digits recognition", font=font_title)
        self.message.grid(column=1, row=0, pady=25)

        #Readme Button
        img_readme = ImageTk.PhotoImage(Image.open("hd_recognition_testing/assets/readme.png").resize((50,50)))
        btn_readme = tkinter.Button(self, image=img_readme, command=lambda: os.startfile("README.md"))
        btn_readme.img = img_readme
        btn_readme.grid(column=2, row=0, padx=60)

        #Button selection frame
        self.btns_frames = tkinter.LabelFrame(self, padx=50, pady=50)
        self.btns_frames.grid(row=1, column=1, columnspan=3, pady=75, padx=(0,170))

        #Menu Buttons
        self.create_model_button = tkinter.Button(self.btns_frames, text="Create a model", font=font_button, command=self.quit)
        self.create_model_button.grid(column=0, row=0, padx=10, pady=10)
        
        self.train_model_button = tkinter.Button(self.btns_frames, text="Train a model", font=font_button, command=self.quit)
        self.train_model_button.grid(column = 1, row = 0, padx=10, pady=10)

        self.evaluate_button = tkinter.Button(self.btns_frames, text="Evaluate", font=font_button, command=self.quit)
        self.evaluate_button.grid(column = 0, row = 1, padx=10, pady=10)
        
        self.predict_button = tkinter.Button(self.btns_frames, text="Predict", font=font_button, command=self.quit)
        self.predict_button.grid(column = 1, row = 1, padx=10, pady=10)
    
    
fenetre = tkinter.Tk()
fenetre.geometry("1280x720")
fenetre.title("Neural Networks")

interface = Interface(fenetre)

interface.mainloop()
interface.destroy()