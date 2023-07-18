import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
import cv2
import pickle
import os

model = load_model('ConvNet/ConvNet-3-conv-128-nodes-2-dense-1689707352.model')
categories = ['Dog', 'Cat']

def prepare(path):
    size = 50
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (size,size))
    img = img.reshape(-1,size,size,1)
    return img

def loop_on_folder(PATH = '/Users/ghaithalmasri/Downloads/PetImages/Dog'):
    correct = 0
    total = 0
    for img in os.listdir(PATH):
        try:
            prediction = model.predict(prepare(os.path.join(PATH,img)))
            if categories[int(prediction[0][0])] == 'Dog':
                correct+=1
                total +=1
            else:
                total += 1     
        except:
            pass
    print(f'Accuracy: {correct/total * 100}')
    return

def quick_inference(image,size):
    img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img,(size,size))
    img = img.reshape(-1,size,size,1)
    prediction = model.predict(img)
    inferenced_image = cv2.imread(image, cv2.IMREAD_COLOR)
    fontFace = cv2.FONT_HERSHEY_SIMPLEX
    org = (50, 100)
    fontScale = 1
    color = (255, 0, 0)
    thickness = 3
    text = categories[int(prediction[0][0])]
    cv2.putText(inferenced_image, str(text), org, fontFace=fontFace, fontScale=fontScale, 
                color=color, thickness=thickness)
    cv2.imshow('Inferenced Image', inferenced_image)
    print('press q to close image')
    cv2.waitKey(0) == 'q'


quick_inference('ConvNet/592.jpg', 50)

