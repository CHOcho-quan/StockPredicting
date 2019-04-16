#-*- coding:utf-8 -*-
import os, csv
ROOT = os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), 'DM Project')
DATAROOT = os.path.join(ROOT, 'data', 'data.csv')

count = 0
with open(DATAROOT, 'r') as f:
    dataset = csv.reader(f)
    for line in dataset:
        #print(line[108], line[109], line[110], line[0], line[1], line[2])
        count += 1

print(count)

