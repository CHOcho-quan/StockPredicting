from sklearn.ensemble import GradientBoostingRegressor
from sklearn.decomposition import PCA
from data_reader import DataLoader
import os
import pickle
import pandas as pd
import numpy as np
import tqdm

N = 10
seq_len = 50
sample_gap = 50

def GBDTRegress(X, y, depth, random_st):
    regr = GradientBoostingRegressor(max_depth=depth, random_state=random_st)
    regr.fit(X, y)
    return regr

if __name__ == '__main__':
    if os.path.exists(path="./data/mydata_{0}_{1}_{2}.pickle".format(N, seq_len, sample_gap)):
        with open("./data/mydata_{0}_{1}_{2}.pickle".format(N, seq_len, sample_gap), 'rb') as load_data:
            (train_input, train_label), (dev_input, dev_label), (test_input, test_label) = pickle.load(load_data)
    else:
        (train_input, train_label), (dev_input, dev_label), (test_input, test_label) = DataLoader("./data.csv", N, seq_len, sample_gap)
        with open("./data/mydata_{0}_{1}_{2}.pickle".format(N, seq_len, sample_gap), 'wb') as save_data:
            data_list = [(train_input, train_label), (dev_input, dev_label), (test_input, test_label)]
            pickle.dump(data_list, save_data)

    pca = PCA(n_components=540)
    train_input = train_input.reshape(train_input.shape[0], -1)
    print(train_input.shape)
    dev_input = dev_input.reshape(dev_input.shape[0], -1)
    test_input = test_input.reshape(test_input.shape[0], -1)
    train_input = pca.fit_transform(train_input)
    dev_input = pca.fit_transform(dev_input)
    test_input = pca.fit_transform(test_input)

    regressor = GBDTRegress(train_input, np.array(train_label), 2, 0)

    pred = regressor.predict(test_input)
    MSE = np.std(pred - np.array(test_label))
    print("MSE LOSS:",MSE)
