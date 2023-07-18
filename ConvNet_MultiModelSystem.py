import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Dense, Flatten
from tensorflow.keras.callbacks import TensorBoard
import pickle
import time


dense_layers = [2]
layer_size = [128]
conv_layers = [3]

X = pickle.load(open('X.pickle', 'rb'))
y = pickle.load(open('y.pickle', 'rb'))


for dense in dense_layers:
    for size in layer_size:
        for conv in conv_layers:
            current = int(time.time())
            NAME = (f'ConvNet-dense{dense}-size{size}-conv{conv}-{current}')
            tensorboard = TensorBoard(log_dir = f'logs/{NAME}')
            model = Sequential()


            model.add(Conv2D(size, (3,3), input_shape = X.shape[1:]))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size = (2,2)))


            for i in range(conv - 1):
                model.add(Conv2D(size, (3,3)))
                model.add(Activation('relu'))
                model.add(MaxPooling2D(pool_size=(2,2)))
            
            model.add(Flatten())
            for i in range(dense):
                model.add(Dense(size))
                model.add(Activation('relu'))
            

            model.add(Dense(1))
            model.add(Activation('sigmoid'))


            model.compile(loss='binary_crossentropy',
                          optimizer = 'adam',
                          metrics = ['accuracy'])
            
            model.fit(X,y, batch_size = 32, epochs = 10, validation_split = 0.3, callbacks = [tensorboard])
            model.save(NAME+'.model')
            