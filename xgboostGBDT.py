import xgboost as xgb
import pandas as pd
import numpy as np
from utils.data_reader import *
import pickle

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

def GBDT(dtrain, dtest, num_round):
    param = {}
    param = {'max_depth': 2, 'eta': 1, 'silent': 1, 'objective': 'binary:logistic'}
    param['nthread'] = 4
    param['eval_metric'] = 'auc'
    evallist = [(dtest, 'eval'), (dtrain, 'train')]
    bst = xgb.train(param, dtrain, num_round, evallist)
    return bst

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
    for batch in train_input:
        batch = np.array(batch)
        for b in batch:
            X_train.append(b)

    print("Generating train label;")
    for batch in train_label:
        batch = np.array(batch)
        for b in batch:
            Y_train.append([b])

    print("Generating validation input;")
    for batch in dev_input:
        batch = np.array(batch)
        X_val.append(batch)

    print("Generating validation label;")
    for batch in dev_label:
        batch = np.array(batch)
        for b in batch:
            Y_val.append([b])

    print("Generating test input;")
    for batch in test_input:
        batch = np.array(batch)
        X_test.append(batch)


    print("Generating test label;")
    for batch in test_label:
        batch = np.array(batch)
        for b in batch:
            Y_test.append([b])

    print(np.array(X_train).shape, np.array(Y_train).shape, np.array(X_val).shape, np.array(Y_val).shape, np.array(X_test).shape, np.array(Y_test).shape)
    dtrain = xgb.DMatrix(data=np.array(X_train).reshape(-1, 1080), label=np.array(Y_train))
    dtest = xgb.DMatrix(data=np.array(X_val).reshape(-1, 1080), label=np.array(Y_val))

    regressor = GBDT(dtrain, dtest, 10)

    pred = regressor.predict(np.array(X_test).reshape(-1, 1080))
    MSE = np.mean(np.square(pred - np.array(Y_test)))
    print(MSE)
    # write_csv(pred)
