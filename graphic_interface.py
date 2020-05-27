
# -*-coding:utf-8 -*

#We will be using the tkinter library
import tkinter

fenetre = tkinter.Tk()
btn_quit = tkinter.Button(fenetre, text="quit", command=fenetre.quit)
btn_quit.pack()
fenetre.mainloop()