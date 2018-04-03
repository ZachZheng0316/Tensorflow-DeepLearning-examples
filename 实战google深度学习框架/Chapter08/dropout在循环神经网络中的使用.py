#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 20:03:07 2017

@author: zach
"""

import tensorflow as tf

lstm_size = 10       # 状态向量的维度(也称隐含层的维度)
number_of_layers = 5 # 深层循环神经网络的层数
batch_size = 5       # batch的大小
number_step = 50     # 数据截断长度

# 定义LSTM结构
lstm = tf.nn.rnn_cell.BasicLSTMCell(lstm_size)

# 使用DropoutWrapper类实现dropout功能。
# 该类通过两个参数来控制dropout的概率
# 一个参数为input_keep_prob，他可以空来控制输入的dropout概率
# 另一个为output_keep_prob，它可以用来控制输出的dropout概率
dropout_lstm = tf.nn.rnn_cell.DropoutWrapper(lstm, output_keep_prob=0.5)

# 在使用了dropout的基础之上定义深层循环神经网络
stacked_lstm = tf.nn.rnn_cell.MultiRNNCell([dropout_lstm] * number_of_layers)

# 通过zero_state函数来获取初始状态
state = stacked_lstm.zero_state(batch_size=batch_size, tf.float32)

# 定义损失变量
loss = 0

# 计算每一个时刻的前向传播结果
for i in range(number_step):
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
    
    
    
    
    
    
    
    
    
    
    
    
    
    