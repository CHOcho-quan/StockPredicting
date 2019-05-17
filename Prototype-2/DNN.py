import os, pickle
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.decomposition import PCA
from data_reader import DataLoader

N = 10
seq_len = 50
sample_gap = 50

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
pca = PCA(n_components=540)
train_input = train_input.reshape(train_input.shape[0], -1)
print(train_input.shape)
dev_input = dev_input.reshape(dev_input.shape[0], -1)
test_input = test_input.reshape(test_input.shape[0], -1)
train_input = pca.fit_transform(train_input)
dev_input = pca.fit_transform(dev_input)
test_input = pca.fit_transform(test_input)

print(train_input.shape)
model = Sequential()
# model.add()
model.add(Dense(128, input_shape=(train_input.shape[1], ), kernel_initializer='uniform', activation='sigmoid', kernel_regularizer=keras.regularizers.l1(0.)))
model.add(Dense(32))
model.add(Dense(8))
model.add(Dense(1))
model.compile(loss="mse", optimizer="adam")
model.summary()

callback = EarlyStopping(monitor="loss", patience=0.01, verbose=1, mode="auto")
model.fit(train_input, train_label, epochs=3, batch_size=32, validation_data=(dev_input, dev_label), callbacks=[callback])

pred = model.predict(test_input)
print(np.std(pred - test_label.reshape(test_label.shape[0], 1)))
print(pred.shape, test_label.shape)
