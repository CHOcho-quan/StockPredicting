import os, pickle
import numpy as np
from sklearn.decomposition import PCA
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from data_reader import DataLoader

N = 50
seq_len = 50
sample_gap = 100

if os.path.exists(path="./data/mydata_{0}_{1}_{2}.pickle".format(N, seq_len, sample_gap)):
    with open("./data/mydata_{0}_{1}_{2}.pickle".format(N, seq_len, sample_gap), 'rb') as load_data:
        (train_input, train_label), (dev_input, dev_label), (test_input, test_label) = pickle.load(load_data)
else:
    (train_input, train_label), (dev_input, dev_label), (test_input, test_label) = DataLoader("./data.csv", N, seq_len, sample_gap)
    with open("./data/mydata_{0}_{1}_{2}.pickle".format(N, seq_len, sample_gap), 'wb') as save_data:
        data_list = [(train_input, train_label), (dev_input, dev_label), (test_input, test_label)]
        pickle.dump(data_list, save_data)

"""
Train input shape N_samples * seq_len * 108
Train label shape N_samples * 1

"""
model = Sequential()
model.add(LSTM(128, input_length=train_input.shape[1], input_dim=train_input.shape[2], return_sequences=True))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(10))
model.add(Dense(1))
model.compile(loss="mse", optimizer="adam")
model.summary()

callback = EarlyStopping(monitor="loss", patience=0.01, verbose=1, mode="auto")
model.fit(train_input, train_label, epochs=3, batch_size=32, validation_data=(dev_input, dev_label), callbacks=[callback])

pred = model.predict(test_input)
print(np.std(pred - test_label.reshape(test_label.shape[0], 1)))
print(pred.shape, test_label.shape)
