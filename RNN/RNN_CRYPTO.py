import pickle
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, BatchNormalization
from keras.callbacks import TensorBoard, ModelCheckpoint
import time


class Rnn_Crypto:
    def __init__(self, epochs: int, LSTM: list, dense: list, size: list, X_train: np.array, y_train: list, X_test: np.array, y_test: list) -> None:
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.size = size
        self.LSTM = LSTM
        self.dense = dense
        self.epochs = epochs

    def train(self):
        X_train = self.X_train
        y_train = self.y_train
        X_test = self.X_test
        y_test = self.y_test
        sizes_list = self.size
        lstm_list = self.LSTM
        dense_list = self.dense
        epochs = self.epochs

        for size in sizes_list:
            for lstm in lstm_list:
                for dense in dense_list:

                    NAME = f'RNN_{lstm}_LSTMs_{size}_size_{dense}_dense_{time.time()}'
                    model = Sequential()

                    model.add(LSTM(size, activation='tanh',
                                   input_shape=X_train.shape[1:], return_sequences=True))
                    model.add(Dropout(0.2))
                    model.add(BatchNormalization())


                    for _ in range(lstm - 2):
                        model.add(LSTM(size, activation='tanh',
                                       return_sequences=True))
                        model.add(Dropout(0.2))
                        model.add(BatchNormalization())



                    model.add(LSTM(size))
                    model.add(Dropout(0.2))
                    model.add(BatchNormalization())


                    for _ in range(dense - 1):
                        model.add(Dense(size, activation='relu'))
                        model.add(Dropout(0.2))

                        

                    model.add(Dense(1, activation='sigmoid'))


                    # Issue with keras 4 new Adam on mac m1 and m2 chips
                    # lr_optim = tf.keras.optimizers.schedules.ExponentialDecay(
                    #     initial_learning_rate=1e-3, decay_steps=100000, decay_rate=0.96,)
                    optimiz = tf.keras.optimizers.legacy.Adam(learning_rate=1e-3, decay = 1e-5)
                    model.compile(optimizer=optimiz,
                                  loss='binary_crossentropy',
                                  metrics=['accuracy'])
                    


                    # setting callbacks
                    filepath = "RNN_Final-{epoch:02d}-{val_accuracy:.3f}"
                    checkpoint = ModelCheckpoint("models/{}.model".format(
                        filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max'))
                    
                    tensorboard = TensorBoard(log_dir=f'logs/{NAME}.model')



                    model.fit(X_train, y_train, batch_size=64,
                              epochs=epochs, validation_data=(X_test, y_test),
                              callbacks=[tensorboard, checkpoint])

                    score = model.evaluate(X_test, y_test, verbose=0)
                    print(f'Test loss: {score[0]}')
                    print(f'Test accuracy: {score[1]}')
                    model.save(f'models/{NAME}.model')


if __name__ == '__main__':
    epochs = 20
    X_train = np.array(pickle.load(open('eth_15min_filtered_data/X_train.pickle','rb')))
    y_train = np.array(pickle.load(open('eth_15min_filtered_data/y_train.pickle','rb')))
    X_test = np.array(pickle.load(open('eth_15min_filtered_data/X_test.pickle','rb')))
    y_test = np.array(pickle.load(open('eth_15min_filtered_data/y_test.pickle','rb')))
    size = [32,64,128, 256, 516]
    dense = [1,2,3]
    lstm = [1,2,3]

    r = Rnn_Crypto(epochs=epochs, LSTM=lstm, dense=dense, size=size,
                   X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
    r.train()

