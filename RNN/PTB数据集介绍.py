# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import tensorflow as tf
from tensorflow.models.rnn.ptb import reader

# 读取数据并打印长度及前100位数据
DATA_PATH = "../datasets/PTB_data"
train_data, valid_data, test_data, _ = reader.ptb_raw_data(DATA_PATH)
print len(train_data)
print train_data[:100]


# 将训练数据组织成batch大小为4、截断长度为5的数据组。
# 并使用队列读取前3个batch。
result = reader.ptb_iterator(train_data, 4, 5)
x, y = result.next()
print "X:", x
print "y:", y


















