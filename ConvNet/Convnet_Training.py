import numpy as np
import pickle
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dense, Flatten
from keras.callbacks import TensorBoard
import time
class Train_Custom_Conv2D:
    def __init__(self, dense_layers:list, conv_layers:list, layer_sizes:list, epochs:int) -> None:
        self.dense_layers = dense_layers
        self.conv_layers = conv_layers
        self.layer_sizes = layer_sizes
        self.X = pickle.load(open('X.pickle','rb'))
        self.y = pickle.load(open('y.pickle','rb'))
        self.X = self.X/255.0
        self.y = np.array(self.y)
        self.epochs = epochs
    
    def train(self):
        dense_layers = self.dense_layers
        conv_layers = self.conv_layers
        layer_sizes = self.layer_sizes


        for dense in dense_layers:
            for conv in conv_layers:
                for size in layer_sizes:
                    model = Sequential()
                    current = time.time()
                    NAME = f'ConvNet-{dense}dense-{conv}conv-{size}size-{current}'
                    tensorboard = TensorBoard(log_dir='logs/'+NAME)

                    model.add(Conv2D(size, (3,3), input_shape = self.X.shape[1:]))
                    model.add(Activation('relu'))
                    model.add(MaxPooling2D((2,2)))


                    secondary_size = size
                    for _ in range(len(conv_layers) - 1):
                        model.add(Conv2D(int(secondary_size),(3,3)))
                        model.add(Activation('relu'))
                        model.add(MaxPooling2D((2,2)))
                        secondary_size = secondary_size/2

                    model.add(Flatten())
                    for _ in range(len(dense_layers)):
                        model.add(Dense(size))
                        model.add(Activation('relu'))

                    model.add(Dense(1))
                    model.add(Activation('sigmoid'))


                    model.compile(optimizer='adam',
                                  loss='binary_crossentropy',
                                  metrics=['accuracy'])
                    model.fit(self.X, self.y, batch_size=32, epochs= self.epochs, validation_split=0.3, callbacks=[tensorboard])



if __name__ == "__main__":
    '''
    
    
    
    To view on tensorboard, start training and cd into the directory that has logs in it
    write tensorboard --logdir logs
    you can open localhost:6006 to view your models training live.
    '''

    dense_layers = [3,4,5] #pass dense layers in a list
    conv_layers = [4,5,6] #pass conv layers in a list
    layer_sizes = [256,512] #pass layer size in a list
    epochs = 10

    c = Train_Custom_Conv2D(dense_layers=dense_layers, conv_layers=conv_layers, layer_sizes=layer_sizes, epochs=epochs)
    c.train()






