from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
import os
import pickle
import pandas as pd
import numpy as np
import tqdm

def splitData(X, y, rate):
    X_train = X[int(X.shape[0]*rate):]
    Y_train = y[int(y.shape[0]*rate):]
    x_val = X[:int(X.shape[0]*rate)]
    y_val = y[:int(y.shape[0]*rate)]
    return X_train, Y_train, x_val, y_val

def write_csv(predict,file_name):
    import csv
    with open(file_name,'w') as f:
        writer = csv.writer(f)
        writer.writerow(
            ["Id", "MidPrice"])
        for i in range(0, len(predict)):
            writer.writerow([(1 + i), predict[i][0]])

def shuffle(X,Y):
    np.random.seed(10)
    randomList = np.arange(X.shape[0])
    np.random.shuffle(randomList)
    return X[randomList], Y[randomList]

def getDataset(path, label_name):
    dataset = pd.read_csv(filepath_or_buffer=path)

    col = dataset.columns.values.tolist()
    data_x = np.array(col[2:-1])
    data_y = data[label_name]

    return data_x, data_y

def RandomForestRegress(X, y, depth, random_st, n_estimators):
    regr = RandomForestRegressor(max_depth=depth, random_state=random_st, n_estimators=n_estimators)
    regr.fit(X, y)
    return regr

if __name__ == '__main__':
    if os.path.exists(path="./data/mydata.pickle"):
        with open('./data/mydata.pickle', 'rb') as load_data:
            (train_input, train_label), (dev_input, dev_label), (test_input, test_label) = pickle.load(load_data)
    else:
        (train_input, train_label), (dev_input, dev_label), (test_input, test_label) = DataLoader("./data/data.csv", N, seq_len, sample_gap, batch_size)
        with open('./data/mydata.pickle', 'wb') as save_data:
            data_list = [(train_input, train_label), (dev_input, dev_label), (test_input, test_label)]
            pickle.dump(data_list, save_data)

    X_train = []
    Y_train = []
    X_val = []
    Y_val = []
    X_test = []
    Y_test = []
    # print(train_label)
    # print(np.array(train_label).shape)
    print("Generating train input;")
    for batch in tqdm.tqdm(train_input):
        batch = np.array(batch)
        for b in batch:
            X_train.append(b)

    print("Generating train label;")
    for batch in tqdm.tqdm(train_label):
        batch = np.array(batch)
        for b in batch:
            Y_train.append(b)

    print("Generating validation input;")
    for batch in tqdm.tqdm(dev_input):
        batch = np.array(batch)
        X_val.append(batch)

    print("Generating validation label;")
    for batch in tqdm.tqdm(dev_label):
        batch = np.array(batch)
        for b in batch:
            Y_val.append(b)

    print("Generating test input;")
    for batch in tqdm.tqdm(test_input):
        batch = np.array(batch)
        X_test.append(batch)


    print("Generating test label;")
    for batch in tqdm.tqdm(test_label):
        batch = np.array(batch)
        for b in batch:
            Y_test.append(b)

    # Implementing PCA on the dataset
    print("Implementing PCA")
    pca = PCA(n_components=78)
    X_train = np.array(X_train).reshape(-1, 108)
    X_train = pca.fit_transform(X_train)
    X_test = np.array(X_test).reshape(-1, 108)
    X_test = pca.fit_transform(X_test)

    # Random Forest regression
    print("Implementing Random Forest Regression")
    regressor = RandomForestRegress(X_train.reshape(-1, 780), np.array(Y_train), 2, 0, 100)

    # Calculating the MSE Loss for the standard
    pred = regressor.predict(X_test.reshape(-1, 780))
    MSE = np.mean(np.square(pred - np.array(Y_test)))
    print("MSE LOSS:",MSE)
    # write_csv(pred)
