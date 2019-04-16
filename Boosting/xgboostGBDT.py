import xgboost as xgb
import pandas as pd
import numpy as np

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
    X, y = getDataset("data.csv", "label")
    X_train, Y_train = shuffle(X_train, Y_train)
    X_train, Y_train, x_val, y_val = splitData(X, y, 0.4)
    dtrain = xgb.DMatrix(data=X_train, label=Y_train)
    dtest = xgb.DMatrix(data=x_val, label=y_val)

    regressor = GBDT(dtrain, detest, 10)

    pred = regressor.predict(x_val)
    write_csv(pred)
