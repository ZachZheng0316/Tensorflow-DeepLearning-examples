#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 19:22:42 2017

@author: zach
"""

import numpy as np

X = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]                 # 输入数据
state = [0.0, 0.0]                                  # 状态数据
w_cell_state = np.asarray([[0.1, 0.2], [0.3, 0.4]]) # 2x2的状态数据的RNN权重
w_cell_input = np.asarray([0.5, 0.6])               # 1x2的输入数据的RNN权重
b_cell = np.asarray([0.1, -0.1])                    # 1x2的RNN偏置项
w_output = np.asarray([[1.0], [2.0]])               # 2x1的输出全连接层
b_output = 0.1                                      # 输出全连接层的偏置项

# 按照时间顺序执行循环神经网络的前向传播过程
for i in range(len(X)):
    # 计算循环体中全连接层神经网络
    before_activation = np.dot(state, w_cell_state) + X[i] * w_cell_input + b_cell
    
    # 更新输入状态
    state = np.tanh(before_activation)
    
    # 根据当前状态计算最终的输出
    final_output = np.dot(state, w_output) + b_output
    
    # 输出每个时刻的信息
    print "befor activation: ", before_activation
    print "sate: ", state
    print "output: ", final_output