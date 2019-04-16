#-*- coding:utf-8 -*-
import os
ROOT = os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), 'CodeFramework')#CodeFramework替换成项目名
DATAROOT = os.path.join(ROOT, 'data', 'dataset')#dataset替换成数据集名称
TRAINROOT = os.path.join(DATAROOT, 'train.json')
DEVROOT = os.path.join(DATAROOT, 'dev.json')
TESTROOT = os.path.join(DATAROOT, 'test.json')