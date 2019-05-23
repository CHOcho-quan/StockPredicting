import featuretools as ft
import pywt
import os, csv, datetime, time, random, pickle
import numpy as np
import pandas as pd
from data_reader import DataLoader

TMP = ''

class Data():
    """
    Loading a single line of the data

    """
    def __init__(self, data):
        self.fv = [float(d) for d in data[:108]]
        self.midprice = float(data[108])
        self.uptime = data[109]
        self.lastprice = float(data[111])
        self.volume = float(data[112])
        self.lastvolume = float(data[113])
        self.turnover = float(data[114])
        self.lastturnover = float(data[115])
        self.askprice = [float(data[120]), float(data[119]), float(data[118]), float(data[117]), float(data[116])]
        self.bidprice = [float(data[121]), float(data[122]), float(data[123]), float(data[124]), float(data[125])]
        self.askvolume = [float(data[130]), float(data[129]), float(data[128]), float(data[127]), float(data[126])]
        self.bibdvolume = [float(data[131]), float(data[132]), float(data[133]), float(data[134]), float(data[135])]
        self.openinterest = float(data[136])
        self.upper = float(data[137])
        self.lower = float(data[138])
        self.day = 0
        self.apm = ''

        self.init_time()

    def get_feature_vector(self):
        return np.array(self.fv)

    def init_time(self):
        """
        Init time of the single line data

        """
        time_digit = self.uptime
        param = time_digit.split(':')
        h = int(param[0])
        m = int(param[1])
        s = int(param[2])
        self.uptime = datetime.time(h, m, s)
        if self.uptime > datetime.time(12,0,0):
            self.apm = 'pm'
        else:
            self.apm = 'am'


def get_fromcsv(root):
    """
    Reading the data from csv simutaniously wash out the data that across two days

    """
    flag = False
    dataset = []
    begin = time.time()
    last_time = datetime.time(0,0,0)
    day_count = 0
    global TMP
    with open(root, 'r') as f:
        datafile = csv.reader(f)
        for line in datafile:
            if flag:
                TMP = line
                data = Data(line)
                dataset.append(data)
                if data.uptime < last_time:
                    day_count += 1
                    data.day = day_count
                    #print(data.day)

                # if day_count == 3:
                #     break

                last_time = data.uptime
            else:
                flag = True
    end = time.time()
    print('dataset length:\t{}\ttime:\t{}'.format(len(dataset), end - begin))
    return dataset

def merge_data(simdatas):
    newdata = Data(TMP)
    l = len(simdatas)
    new_fv = [0 for i in range(108)]
    new_midprice = 0
    new_lastvolume = 0
    new_lastturnover = 0
    new_askprice = [0 for i in range(5)]
    new_bidprice = [0 for i in range(5)]
    new_askvolume = [0 for i in range(5)]
    new_bidvolume = [0 for i in range(5)]
    lastdata = simdatas[-1]
    for data in simdatas:
        for i in range(108):
            new_fv[i] += data.fv[i]
        new_midprice += data.midprice
        new_lastvolume += data.lastvolume
        new_lastturnover += data.lastturnover
        for i in range(5):
            new_askprice[i] += data.askprice[i]
            new_bidprice[i] += data.bidprice[i]
            new_askvolume[i] += data.askvolume[i]
            new_bidvolume[i] += data.bibdvolume[i]

    for i in range(108):
        new_fv[i] /= l

    new_midprice /= l
    new_lastvolume /= l
    new_lastturnover /= l

    for i in range(5):
        new_askprice[i] /= l
        new_bidprice[i] /= l
        new_askvolume[i] /= l
        new_bidvolume[i] /= l

    newdata.fv = new_fv
    newdata.midprice = new_midprice
    newdata.uptime = lastdata.uptime
    newdata.lastprice = lastdata.lastprice
    newdata.volume = lastdata.volume
    newdata.lastvolume = new_lastvolume
    newdata.turnover = lastdata.turnover
    newdata.lastturnover = new_lastturnover
    newdata.askprice = new_askprice
    newdata.bidprice = new_bidprice
    newdata.askvolume = new_askvolume
    newdata.bibdvolume = new_bidvolume
    newdata.openinterest = lastdata.openinterest
    newdata.upper = lastdata.upper
    newdata.lower = lastdata.lower
    newdata.day = lastdata.day
    newdata.apm = lastdata.apm

    return newdata

def clean_data(dataset):
    """
    Wash out the data that is at the same time

    """
    last_time = dataset[0].uptime
    simdatas = []
    new_dataset = []
    for data in dataset:
        if data.uptime == last_time:
            simdatas.append(data)
        else:
            new_dataset.append(merge_data(simdatas))
            simdatas = [data]
            last_time = data.uptime

    if len(simdatas) > 0:
        new_dataset.append(merge_data(simdatas))

    return new_dataset

def get_feature_label(new_dataset, n, seq_len, sample_gap):
    data_day_order = []
    tmp = []
    apm = 'am'
    dataset = []
    labels = []
    for data in new_dataset:
        if data.apm == apm:
            tmp.append(data)
        else:
            data_day_order.append(tmp)
            tmp = [data]
            apm = 'pm' if apm == 'am' else 'am'

    for i, data_batch in enumerate(data_day_order):
        data_day_order[i] = data_batch[:-1 * n]
        sample_num = (len(data_day_order[i]) // sample_gap) - 1
        for j in range(sample_num):
            left = j * sample_gap
            mid = left + seq_len
            right = left + seq_len + n
            tmp = []
            for k in range(left, mid):
                tmp.append(data_day_order[i][k].midprice)
                tmp.append(data_day_order[i][k].lastprice)
                tmp.append(data_day_order[i][k].turnover / (data_day_order[i][k].volume + 1e-8))
                tmp.append(data_day_order[i][k].lastturnover / (data_day_order[i][k].lastvolume + 1e-8))
                tmp.append(data_day_order[i][k].upper)
                tmp.append(data_day_order[i][k].lower)
                tmp.extend(data_day_order[i][k].askprice)
                tmp.extend(data_day_order[i][k].bidprice)
                tmp.extend(np.array(data_day_order[i][k].askvolume) - np.array(data_day_order[i][k].bibdvolume))
            dataset.append(tmp)
            current = data_day_order[i][mid - 1].askprice[0] + data_day_order[i][mid - 1].bidprice[0]
            future = data_day_order[i][right - 1].askprice[0] + data_day_order[i][right - 1].bidprice[0]
            label = (future - current) / 2
            labels.append(label)

    print('total data:\t{}\ttotal labels:\t{}'.format(len(dataset), len(labels)))

    return dataset, labels

def divid_dataset(dataset, labels):
    """
    Divide the dataset into train, validation & test

    """
    l = len(dataset)
    idx_list = list(range(l))
    random.shuffle(idx_list)

    train_range = int(l * 0.8)
    dev_range = int(l * 0.9)

    train_input = []
    train_label = []
    dev_input = []
    dev_label = []
    test_input = []
    test_label = []

    for i in range(train_range):
        idx = idx_list[i]
        train_input.append(dataset[idx])
        train_label.append(labels[idx])

    for i in range(train_range, dev_range):
        idx = idx_list[i]
        dev_input.append(dataset[idx])
        dev_label.append(labels[idx])

    for i in range(dev_range, l):
        idx = idx_list[i]
        test_input.append(dataset[idx])
        test_label.append(labels[idx])

    print('train set size:\t{}\tdevelop set size:\t{}\ttest set size:\t{}'.format(len(train_input), len(dev_input), len(test_input)))

    return (train_input, train_label), (dev_input, dev_label), (test_input, test_label)

def generate_by_linearcomb(root, N, seq_len, sample_gap):
    """
    Generating features by linear combination such as addition, subtraction & multiplication

    """
    if os.path.exists(path="./raw_data/featuredata_{0}_{1}_{2}.pickle".format(N, seq_len, sample_gap)):
        with open("./raw_data/featuredata_{0}_{1}_{2}.pickle".format(N, seq_len, sample_gap), 'rb') as load_data:
            (train_input, train_label), (dev_input, dev_label), (test_input, test_label) = pickle.load(load_data)
    else:
        dataset = get_fromcsv(root)
        dataset = clean_data(dataset)
        dataset, labels = get_feature_label(dataset, N, seq_len, sample_gap)
        (train_input, train_label), (dev_input, dev_label), (test_input, test_label) = divid_dataset(dataset, labels)
        with open("./raw_data/featuredata_{0}_{1}_{2}.pickle".format(N, seq_len, sample_gap), 'wb') as save_data:
            data_list = [(np.array(train_input), np.array(train_label)), (np.array(dev_input), np.array(dev_label)), (np.array(test_input), np.array(test_label))]
            pickle.dump(data_list, save_data)

    return (np.array(train_input), np.array(train_label)), (np.array(dev_input), np.array(dev_label)), (np.array(test_input), np.array(test_label))

def wavelet_denoising(data):
    """
    Doing wavelet transformation for single data

    """
    db4 = pywt.Wavelet('db4')
    coeffs = pywt.wavedec(data, db4)
    coeffs[len(coeffs) - 1] *= 0
    coeffs[len(coeffs) - 2] *= 0
    meta = pywt.waverec(coeffs, db4)
    return meta

def generate_by_wavelet(N, seq_len, sample_gap):
    """
    Generating features by wavelet transformation

    """
    if os.path.exists(path="./raw_data/mydata_{0}_{1}_{2}.pickle".format(N, seq_len, sample_gap)):
        with open("./raw_data/mydata_{0}_{1}_{2}.pickle".format(N, seq_len, sample_gap), 'rb') as load_data:
            (train_input, train_label), (dev_input, dev_label), (test_input, test_label) = pickle.load(load_data)
    else:
        (train_input, train_label), (dev_input, dev_label), (test_input, test_label) = DataLoader("./data.csv", N, seq_len, sample_gap)
        with open("./raw_data/mydata_{0}_{1}_{2}.pickle".format(N, seq_len, sample_gap), 'wb') as save_data:
            data_list = [(train_input, train_label), (dev_input, dev_label), (test_input, test_label)]
            pickle.dump(data_list, save_data)

    for i in range(train_input.shape[2]):
        train_input[:, :, i] = wavelet_denoising(train_input[:, :, i])
        dev_input[:, :, i] = wavelet_denoising(dev_input[:, :, i])
        test_input[:, :, i] = wavelet_denoising(test_input[:, :, i])

    return train_input, dev_input, test_input

def get_ft_features(askprice, bidprice, askvolume, bidvolume, others):
    es_train = ft.EntitySet(id='stock')

    es_train = es_train.entity_from_dataframe(entity_id='askprices', dataframe=askprice, index='stock_id', make_index=True)
    es_train = es_train.entity_from_dataframe(entity_id='bidprices', dataframe=bidprice, index='stock_id', make_index=True)
    es_train = es_train.entity_from_dataframe(entity_id='askvolumes', dataframe=askvolume, index='stock_id', make_index=True)
    es_train = es_train.entity_from_dataframe(entity_id='bidvolumes', dataframe=bidvolume, index='stock_id', make_index=True)
    es_train = es_train.entity_from_dataframe(entity_id='otherprices', dataframe=others, index='stock_id', make_index=True)

    r1 = ft.Relationship(es_train['askprices']['stock_id'], es_train['askvolumes']['stock_id'])
    r2 = ft.Relationship(es_train['bidprices']['stock_id'], es_train['bidvolumes']['stock_id'])
    r3 = ft.Relationship(es_train['askprices']['stock_id'], es_train['otherprices']['stock_id'])

    es_train = es_train.add_relationship(r1)
    es_train = es_train.add_relationship(r2)
    es_train = es_train.add_relationship(r3)
    print(es_train)

    features, feature_names = ft.dfs(entityset=es_train, target_entity='askprices')
    print(features)

    return np.array(features)

def generate_by_ft(N, seq_len, sample_gap):
    """
    Generating features by featuretools

    """
    if os.path.exists(path="./raw_data/mydata_{0}_{1}_{2}.pickle".format(N, seq_len, sample_gap)):
        with open("./raw_data/mydata_{0}_{1}_{2}.pickle".format(N, seq_len, sample_gap), 'rb') as load_data:
            (train_input, train_label), (dev_input, dev_label), (test_input, test_label) = pickle.load(load_data)
    else:
        (train_input, train_label), (dev_input, dev_label), (test_input, test_label) = DataLoader("./data.csv", N, seq_len, sample_gap)
        with open("./raw_data/mydata_{0}_{1}_{2}.pickle".format(N, seq_len, sample_gap), 'wb') as save_data:
            data_list = [(train_input, train_label), (dev_input, dev_label), (test_input, test_label)]
            pickle.dump(data_list, save_data)
    train_input = np.array(train_input).reshape(train_input.shape[0], seq_len, -1)
    train_label = np.array(train_label)
    dev_input = np.array(dev_input).reshape(dev_input.shape[0], seq_len, -1)
    dev_label = np.array(dev_label)
    test_input = np.array(test_input).reshape(test_input.shape[0], seq_len, -1)
    test_label = np.array(test_label)
    print(train_input.shape, dev_input.shape, test_input.shape)
    train_df = pd.DataFrame(train_input.reshape(-1, train_input.shape[2]))
    dev_df = pd.DataFrame(dev_input.reshape(-1, dev_input.shape[2]))
    test_df = pd.DataFrame(test_input.reshape(-1, test_input.shape[2]))
    train_df.columns = ['midprice', 'lastprice', 'volume', 'lastvolume', 'turnover', 'lastturnover', 'upper', 'lower',\
                        'askprice5', 'askprice4', 'askprice3', 'askprice2', 'askprice1', 'bidprice1', 'bidprice2', 'bidprice3', 'bidprice4', 'bidprice5', \
                        'askvolume5', 'askvolume4', 'askvolume3', 'askvolume2', 'askvolume1', 'bidvolume1', 'bidvolume2', 'bidvolume3', 'bidvolume4', 'bidvolume5']
    dev_df.columns = ['midprice', 'lastprice', 'volume', 'lastvolume', 'turnover', 'lastturnover', 'upper', 'lower',\
                        'askprice5', 'askprice4', 'askprice3', 'askprice2', 'askprice1', 'bidprice1', 'bidprice2', 'bidprice3', 'bidprice4', 'bidprice5', \
                        'askvolume5', 'askvolume4', 'askvolume3', 'askvolume2', 'askvolume1', 'bidvolume1', 'bidvolume2', 'bidvolume3', 'bidvolume4', 'bidvolume5']
    test_df.columns = ['midprice', 'lastprice', 'volume', 'lastvolume', 'turnover', 'lastturnover', 'upper', 'lower',\
                        'askprice5', 'askprice4', 'askprice3', 'askprice2', 'askprice1', 'bidprice1', 'bidprice2', 'bidprice3', 'bidprice4', 'bidprice5', \
                        'askvolume5', 'askvolume4', 'askvolume3', 'askvolume2', 'askvolume1', 'bidvolume1', 'bidvolume2', 'bidvolume3', 'bidvolume4', 'bidvolume5']

    bidvolume = train_df.iloc[:, 23:]
    bidvolume_dev = dev_df.iloc[:, 23:]
    bidvolume_test = test_df.iloc[:, 23:]

    askvolume = train_df.iloc[:, 18:23]
    askvolume_dev = dev_df.iloc[:, 18:23]
    askvolume_test = test_df.iloc[:, 18:23]

    bidprice = train_df.iloc[:, 13:18]
    bidprice_dev = dev_df.iloc[:, 13:18]
    bidprice_test = test_df.iloc[:, 13:18]

    askprice = train_df.iloc[:, 8:13]
    askprice_dev = dev_df.iloc[:, 8:13]
    askprice_test = test_df.iloc[:, 8:13]

    others = train_df.iloc[:, 0:8]
    others_dev = dev_df.iloc[:, 0:8]
    others_test = test_df.iloc[:, 0:8]

    train_input = get_ft_features(askprice, bidprice, askvolume, bidvolume, others)
    dev_input = get_ft_features(askprice_dev, bidprice_dev, askvolume_dev, bidvolume_dev, others_dev)
    test_input = get_ft_features(askprice_test, bidprice_test, askvolume_test, bidvolume_test, others_test)
    print(train_input.shape, dev_input.shape, test_input.shape)

    with open("./raw_data/ftdata.pickle".format(N, seq_len, sample_gap), 'wb') as save_data:
        data_list = [(train_input, train_label), (dev_input, dev_label), (test_input, test_label)]
        pickle.dump(data_list, save_data)

if os.path.exists(path="./raw_data/ftdata.pickle"):
    with open("./raw_data/ftdata.pickle", 'rb') as load_data:
        (train_input, train_label), (dev_input, dev_label), (test_input, test_label) = pickle.load(load_data)
else:
    (train_input, train_label), (dev_input, dev_label), (test_input, test_label) = DataLoader("./data.csv", N, seq_len, sample_gap)
    with open("./raw_data/ftdata.pickle", 'wb') as save_data:
        data_list = [(train_input, train_label), (dev_input, dev_label), (test_input, test_label)]
        pickle.dump(data_list, save_data)

print(train_input.shape)
train_input = train_input[~np.isnan(train_input)].reshape(int(train_input.shape[0] / 30), -1)
print(train_input.shape)
train_label = np.array(train_label)
dev_input = np.array(dev_input).reshape(-1, 30, dev_input.shape[1])
dev_label = np.array(dev_label)
test_input = np.array(test_input).reshape(-1, 30, test_input.shape[1])
test_label = np.array(test_label)
