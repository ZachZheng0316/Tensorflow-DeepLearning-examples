#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 20:42:22 2017

@author: zach
"""

import numpy as np
import tensorflow as tf
import tensorflow.models.rnn.ptb import reader

# 定义相关参数
DATA_PATH = "../datasets/PTB_data"
HIDDEN_SIZE = 20        # 隐含规模
NUM_LAYERS = 2          # 深层RNN中LSTM结构的层数
VOCAB_SIZE = 10000      # 单词规模，加上语句结束表示符和稀有单词标识符总共1W个单词

LEARNING_RATE = 1.0     # 学习速率
TRAIN_BATCH_SIZE = 20   # 训练数据batch的大小
TRAIN_NUM_STEP = 35     # 训练数据截断长度

# 在测试的时候，不需要使用截断，所以可以将测试数据看成一个超长的序列
EVAL_BATCH_SIZE = 1     # 测试数据batch的大小
EVAL_NUM_STEP = 1       # 测试数据截断长度
NUM_EPOCH = 2           # 使用训练数据的轮数
KEEP_PROB = 0.5         # 节点不被dropout的概率
MAX_GRAD_NORM = 5       # 用于控制梯度膨胀的参数

# 定义一个类来描述模型结构
class PTBModel(object):
    def __init__(self, is_training, batch_size, num_steps):
        # 记录使用batch的大小和长度
        self.batch_size = batch_size    # 记录batch的大小
        self.num_steps = num_steps      # 记录batch的截断长度
        
        # 定义输入层。
        self.input_data = tf.placeholder(tf.int32, [batch_size, num_steps])
        self.targets = tf.placeholder(tf.int32, [batch_size, num_steps])
        
        # 定义使用LSTM结构及训练时使用dropout。
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(HIDDEN_SIZE)
        if is_training:
            lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=KEEP_PROB)
        cell = tf.contrib.rnn.MultiRNNCell([lstm_cell]*NUM_LAYERS)
        
        # 初始化最初的状态。
        self.initial_state = cell.zero_state(batch_size, tf.float32)
        embedding = tf.get_variable("embedding", [VOCAB_SIZE, HIDDEN_SIZE])
        
        # 将原本单词ID转为单词向量。
        inputs = tf.nn.embedding_lookup(embedding, self.input_data)
        
        if is_training:
            inputs = tf.nn.dropout(inputs, KEEP_PROB)

        # 定义输出列表。
        outputs = []
        state = self.initial_state
        with tf.variable_scope("RNN"):
            for time_step in range(num_steps):
                if time_step > 0: tf.get_variable_scope().reuse_variables()
                cell_output, state = cell(inputs[:, time_step, :], state)
                outputs.append(cell_output) 
        output = tf.reshape(tf.concat(outputs, 1), [-1, HIDDEN_SIZE])
        weight = tf.get_variable("weight", [HIDDEN_SIZE, VOCAB_SIZE])
        bias = tf.get_variable("bias", [VOCAB_SIZE])
        logits = tf.matmul(output, weight) + bias
        
        # 定义交叉熵损失函数和平均损失。
        loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
            [logits],
            [tf.reshape(self.targets, [-1])],
            [tf.ones([batch_size * num_steps], dtype=tf.float32)])
        self.cost = tf.reduce_sum(loss) / batch_size
        self.final_state = state
        
        # 只在训练模型时定义反向传播操作。
        if not is_training: return
        trainable_variables = tf.trainable_variables()

        # 控制梯度大小，定义优化方法和训练步骤。
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, trainable_variables), MAX_GRAD_NORM)
        optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE)
        self.train_op = optimizer.apply_gradients(zip(grads, trainable_variables))
