#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 19:49:45 2017

@author: zach
"""

import tensorflow as tf

lstm_size = 10       # lstm隐含层的大小
number_of_layers = 5 # 深层循环神经网络的层数
batch_size = 50      # 截断长度

# 定义一个基本的LSTM结构作为循环体的基本结构。
# 深层循环神经网络也支持其他循环体结构
lstm = tf.nn.rnn_cell.BasicLSTMCell(lstm_size)

# 通过MultiRNNCell类实现深层循环神经网络中每一个时刻的前向传播过程
# 其中，number_of_layers表示有多少层
stacked_lstm = tf.nn.rnn_cell.MultiRNNCell([lstm] * number_of_layers)

# 通过zeros_state函数来获取初始状态。
# 和经典的循环神经网络一样
state = stacked_lstm.zero_state(batch_size=batch_size, tf.float32)

# 计算每一时刻前向传播结果
# 在训练的过程中为了避免梯度消散的问题，会规定一个最大的序列长度
for i in range(len(num_steps)):
    # 在第一个时刻声明LSTM结构中使用变量，在之后的时刻都需要复用之前定义好的变量
    if i > 0:
        tf.get_variable_scope().reuse_variables()
    
    # 每一步处理时间序列中的一个时刻。
    # 将当前输入(current_input)和前一个时刻的(state)传入定义好的LSTM结构
    # 得到当前LSTM结构的输出lstm_output和更新后的状态state
    stacked_lstm_output, state = stacked_lstm(current_input, state)
    
    # 将当前时刻LSTM结构的输出传入一个全连接层得到最后的输出
    final_output = fully_connected(stacked_lstm_output)
    
    # 计算当前时刻输出的损失
    loss += calc_loss(final_output, expected_output)