import tensorflow as tf
from keras.models import Model
from keras.layers import GRU, Dense, LayerNormalization, Input, LSTM, Dropout, BatchNormalization
from keras.callbacks import ModelCheckpoint, TensorBoard
import pickle
import numpy as np
import time
import keras

NAME = f'GRU_6_LAYERS_LIKE_PAPER_{time.time()}'
X_train = np.array(pickle.load(open('eth_15min_filtered_data/X_train.pickle','rb')))
y_train = np.array(pickle.load(open('eth_15min_filtered_data/y_train.pickle','rb')))
X_test = np.array(pickle.load(open('eth_15min_filtered_data/X_test.pickle','rb')))
y_test = np.array(pickle.load(open('eth_15min_filtered_data/y_test.pickle','rb')))

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)



model = keras.Sequential()
model.add(keras.layers.LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(keras.layers.Dense(32, activation='relu'))
model.add(keras.layers.Dense(1, activation='sigmoid'))



optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.0001)

tensorboard= TensorBoard(log_dir=f'logs/{NAME}.model')


filepath = "GRU-final"
checkpoint = ModelCheckpoint("models/{}.model".format(
                        filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max'))


model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

if __name__ == "__main__":



    epochs = 20

    model.fit(X_train, y_train, batch_size=32, epochs=epochs, validation_data=(X_test, y_test), callbacks=[tensorboard, checkpoint])
