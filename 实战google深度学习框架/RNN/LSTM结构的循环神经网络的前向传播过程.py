#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 07:56:48 2017

@author: zach
"""

import tensorflow as tf
import numpy as np

# 定义数据
X = [1, 2]
Y = [8, 7]

HIDDEN_SIZE = 200   # 定义隐含层的规模
batch_size = 1      # 定义batch的大小
num_steps = 2       # 定义数据截断的长度

# 定义一个LSTM结构
lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE)

# 将LSTM中的状态初始化为全0的数组
state = lstm_cell.zero_state(batch_size, tf.float32)

# 定义用于输出的全连接层
w_output = np.asarray([[1.0], [2.0]])
b_output = 0.1

# 定义损失函数
loss = 0.0


# 按时间顺序循环执行神经网络的前向传播过程
for i in range(num_steps):
    # 在第一个时刻声明LSTM结构中使用的变量，在这之后的时刻都需要复用之前定义好的变量
    if i > 0:
        tf.get_variable_scope().reuse_variables()
    
    # 每一步处理时间序列中的一个时刻。
    # 将当前输入(current_input)和前一个时刻状态(state)传入定义的LSTM结构
    # 得到当前LSTM结构的输出lstm_output和更新后的状态state
    lstm_output, state = lstm_cell(X[i], state) # ===>这个地发有问题
    
    # 将当前时刻LSTM结构的输出传入一个全连接层得到哦最后的输出
    #final_output = fully_connected(lstm_output)
    final_output = np.dot(state, w_output) + b_output
    
    # 计算当前时刻的损失
    #loss += calc_loss(final_output, expected_output)
    
    # 输出每个时刻的信息
    print "X: ", X[i]
    print "state: ", state
    print "output: ", final_output
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    