import sklearn.ensemble import RandomForestRegressor
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

def RandomForestRegress(X, y, depth, random_st, n_estimators):
    regr = RandomForestRegressor(max_depth=depth, random_state=random_st, n_estimators=n_estimators)
    regr.fit(X, y)
    return regr

if __name__ == '__main__':
    X, y = getDataset("data.csv", "label")
    X_train, Y_train = shuffle(X_train, Y_train)
    X_train, Y_train, x_val, y_val = splitData(X, y, 0.4)

    regressor = RandomForestRegress(X_train, Y_train, 2, 0, 100)

    pred = regressor.predict(X_train)
    write_csv(pred)
