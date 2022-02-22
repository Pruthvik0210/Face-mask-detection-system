from tkinter import *
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from flask import Flask,render_template,request
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename
import cv2
import matplotlib.pyplot as plt
import cv2 as cv
import tensorflow as tf


# load the model
model=tf.keras.models.load_model('my_model.h5')
images ="test_images"
caffeModel = "models\\res10_300x300_ssd_iter_140000.caffemodel"
prototextPath = "models\\deploy.prototxt"
cvNet = cv.dnn.readNetFromCaffe(prototextPath,caffeModel)
def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)])
    return cv2.LUT(image.astype(np.uint8), table.astype(np.uint8))

def model_predict(img_path,model):
    images ="test_images"
    img_size = 124
    gamma = 2.0
    fig = plt.figure(figsize = (14,14))
    rows = 3
    cols = 2
    axes = []
    assign = {'0':'Mask','1':"No Mask"}
    image =  cv2.imread(os.path.join(images,img_path),1)
    image =  adjust_gamma(image, gamma=gamma)
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300,300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    cvNet.setInput(blob)
    detections = cvNet.forward()
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.2:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            frame = image[startY:endY, startX:endX]
            im = cv2.resize(frame,(img_size,img_size))
            im = np.array(im)/255.0
            im = im.reshape(1,124,124,3)
            result = model.predict(im)
            if result>0.5:
                label_Y = 1
            else:
                label_Y = 0

    
            cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
            cv2.putText(image,assign[str(label_Y)] , (startX, startY-10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (36,255,12), 2)
    
    #axes.append(fig.add_subplot(rows, cols))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()
    return label_Y

root = Tk()
root.title("Face Mask Detector")

root.maxsize(900, 600)
root.config(bg="Black")

def get_url():
    img_url = size_entry.get()
    return img_url

def run_img():
    global img_url
    global image1
    global test
    Label(Mainframe, text= "                 ", bg='Grey').grid(row=1, column=1, padx=5, pady=5, sticky=W)
    img_url = get_url()
    #print(img_url)
    image =  os.path.join(images,img_url)
    w=580
    h=360
    image1 = Image.open(image)
    resize_image = image1.resize((w, h))
    test = ImageTk.PhotoImage(resize_image)
    Label(canvas, image=test, bg='Grey').grid(row=0, column=0, padx=5, pady=5)
    result = model_predict(img_url,model)
#     print(result))

Mainframe = Frame(root, width=600, height=200, bg="Grey")
Mainframe.grid(row=0, column=0, padx=10, pady=5)
canvas = Canvas(root, width=600, height=380, bg="Grey")
canvas.grid(row=1, column=0, padx=10, pady=5)


size_entry = Entry(Mainframe)
size_entry.grid(row=0, column=1, padx=5, pady=5) 
size_entry.get()

B1 = Button(Mainframe, text="Predict", bg="Blue", command=run_img).grid(row=0, column=6, padx=5, pady=5)
Label(Mainframe, text="Enter URL", bg='Grey').grid(row=0, column=0, padx=5, pady=5, sticky=W)    

root.mainloop()