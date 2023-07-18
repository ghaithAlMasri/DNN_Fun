import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
import cv2
import pickle
import os
model = load_model('ConvNet-3-conv-128-nodes-2-dense-1689707352.model')


categories = ['Dog', 'Cat']
def prepare(path):
    size = 50
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (size,size))
    img = img.reshape(-1,size,size,1)
    return img


PATH = '/Users/ghaithalmasri/Downloads/PetImages/Dog'
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

