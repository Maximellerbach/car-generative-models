import time
from tkinter import Button, DoubleVar, Entry, Scale, StringVar, Tk

import cv2
import numpy as np
from tensorflow.keras.models import load_model

gen = load_model("vroum\\vroumgen.h5")


def pred(a=None):
    noise = []
    for c in var_list:
        noise.append((c.get() - 50) / 50)

    x = gen(np.expand_dims(np.array(noise), axis=0)).numpy()

    cv2.imshow("img", (x[0] + 1) * 0.5)
    cv2.waitKey(1)


def randomize():
    for c in var_list:
        # c.set(np.random.randint(0, 100))
        r = np.random.normal(loc=50, scale=50)
        c.set(r)
    pred()


def save():
    noise = []
    for c in var_list:
        noise.append((c.get() - 50) / 50)

    np.save("gen\\" + str(time.time()) + ".npy", np.array(noise))


def load():
    no = np.load("gen\\" + string.get())
    for it, c in enumerate(var_list):
        c.set((no[it] + 1) * 50)
    pred()


root = Tk()

scale_list = []
var_list = []

for i in range(100):
    var = DoubleVar()
    var.set(0)

    scale = Scale(root, variable=var, command=pred).grid(row=i // 10, column=i % 10)
    var_list.append(var)
    scale_list.append(scale)

randomize()

button = Button(root, text="Get Scale Value", command=pred).grid(row=9, column=10)
button = Button(root, text="randomize", command=randomize).grid(row=8, column=10)
button = Button(root, text="save", command=save).grid(row=7, column=10)
string = StringVar()
entree = Entry(root, textvariable=string, width=30).grid(row=6, column=10)
button = Button(root, text="load", command=load).grid(row=5, column=10)
pred()

root.mainloop()
