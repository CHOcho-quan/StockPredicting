#-*- coding:utf-8 -*-
import os, csv, datetime, time, random
import torch
import numpy as np

TMP = ''

class Data():
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
        return torch.Tensor(self.fv)

    def init_time(self):
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
                    print(data.day)

                if day_count == 3:
                    break

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

def get_label(new_dataset, n, seq_len, sample_gap):
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
                tmp.append(data_day_order[i][k].fv)
            dataset.append(tmp)
            current = data_day_order[i][mid - 1].askprice[0] + data_day_order[i][mid - 1].bidprice[0]
            future = data_day_order[i][right - 1].askprice[0] + data_day_order[i][right - 1].bidprice[0]
            label = (future - current) / 2
            labels.append(label)

    print('total data:\t{}\ttotal labels:\t{}'.format(len(dataset), len(labels)))

    return dataset, labels

def divid_dataset(dataset, labels):
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


def get_minibatch(dataset, labels, batch_size, seq_len):
    count = 0
    batch_data = []
    batch_label = []
    mini_batch_data = []
    mini_batch_label = []
    for idx, data in enumerate(dataset):
        mini_batch_data.append(data)
        mini_batch_label.append(labels[idx])
        count += 1
        if count == batch_size:
            mini_batch_data = normalize_minibatch(mini_batch_data, seq_len)
            batch_data.append(mini_batch_data)
            batch_label.append(torch.Tensor(mini_batch_label))
            mini_batch_data = []
            mini_batch_label = []
            count = 0
    return batch_data, batch_label


def normalize_minibatch(mini_batch, seq_len):
    batch_data = [[] for i in range(seq_len)]
    for data in mini_batch:
        for i in range(seq_len):
            batch_data[i].append(data[i])
    batch_data = torch.Tensor(batch_data)

    return batch_data

def DataLoader(root, N, seq_len, sample_gap, batch_size, dev_bs = 1, test_bs = 1):
    dataset = get_fromcsv(root)
    dataset = clean_data(dataset)
    dataset, labels = get_label(dataset, N, seq_len, sample_gap)
    (train_input, train_label), (dev_input, dev_label), (test_input, test_label) = divid_dataset(dataset, labels)
    train_input, train_label = get_minibatch(train_input, train_label, batch_size, seq_len)
    dev_input, dev_label = get_minibatch(dev_input, dev_label, dev_bs, seq_len)
    test_input, test_label = get_minibatch(test_input, test_label, test_bs, seq_len)
    return (train_input, train_label), (dev_input, dev_label), (test_input, test_label)




if __name__ == '__main__':
    ROOT = os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), 'DM Project')
    DATAROOT = os.path.join(ROOT, 'data', 'data.csv')
    (train_input, train_label), (dev_input, dev_label), (test_input, test_label) = DataLoader(DATAROOT, 10, 10, 10, 32)
    raise ValueError