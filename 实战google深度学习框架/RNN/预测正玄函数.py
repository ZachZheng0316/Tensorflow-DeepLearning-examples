#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 19:42:00 2017

@author: zach
"""

import numpy as np
import tensorflow as tf
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt

learn = tf.contrib.learn

HIDDEN_SIZE = 30            # lstm中隐含的节点个数
NUM_LAYERS = 2              # lstm的层数
TIMESTEPS = 10              # 循环神经网络的截断长度
TRAING_STEPS = 3000        # 训练的轮数
BATCH_SIZE = 32             # batch的大小
TRAINING_EXAMPLES = 10000   # 训练数据的个数
TESTING_EXAMPLES = 1000     # 测试数据的个数
SAMPLE_GAP = 0.01           # 采样间隔

# 产生数据集
def generate_data(seq):
    '''
    参数意义
        seq : 表示序列
    函数意义：
        根据提供的序列，产生样本训练集和预测值
    '''
    X = []
    y = []
    
    # 序列的第i项和后面的TIMESTEPS-1项合在一起作为输入
    # 第i+TIMESTEPS作为输出。
    # 即用sin函数前面的TIMESTEPS个点的信息，预测第i+TIMESTEPS的点的函数值
    for i in range(len(seq) - TIMESTEPS - 1):
        X.append([seq[i:i+TIMESTEPS]])
        y.append([seq[i + TIMESTEPS]])
    
    # 返回X和y
    return np.array(X), np.array(y)
    
    
# 建立LSTM模型
def lstm_model(X, y):
    '''
    参数意义：
        X: 输入的样本集
        y: 样本集的预测值
    函数意义：
        
    '''
    
    # 使用多层的LSTM结构
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(HIDDEN_SIZE, state_is_tuple=True) # 建立LSTM结构
    cell = tf.contrib.rnn.MultiRNNCell([lstm_cell] * NUM_LAYERS)  # 建立多层RNN模型  
    
    output, _ = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
    output = tf.reshape(output, [-1, HIDDEN_SIZE])
    
    # 
    prediction, loss = learn.models.linear_regression(output, y)
    
    # 创建模型优化器并得到优化的步骤
    train_op = tf.contrib.layers.optimize_loss(
            loss,                                   #  损失函数
            tf.contrib.framework.get_global_step(), #  获取训练步数，并在训练的时候更新
            optimizer='Adagrad',                    #  优化方法
            learning_rate=0.1)                      #  学习速率
    
    # 返回预测值、损失值、优化操作
    return prediction, loss, train_op
    

# 用正玄函数生成训练数据集和测试数据集
# numpy.linspace函数可以创建一个等差序列的数组，它常用的参数有三个参数
# 第一个参数表示起始值，第二个参数表示终止值，第三个参数表示数列的长度。
# 例如：linespace(1, 10, 10)产生的数组为arrray([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
test_start = TRAINING_EXAMPLES * SAMPLE_GAP 
test_end = (TRAINING_EXAMPLES + TESTING_EXAMPLES) * SAMPLE_GAP
# train_X的shape为[9989, 1, 10], train_y的shape为[9989, 10]
train_X, train_y = generate_data(np.sin(np.linspace(0, test_start, TRAINING_EXAMPLES, dtype=np.float32)))
# test_X's shape:[989, 10], test_y's shape:[989, 1]
test_X, test_y = generate_data(np.sin(np.linspace(test_start, test_end, TESTING_EXAMPLES, dtype=np.float32)))


# 建立深层循环神经网络
regressor = learn.Estimator(model_fn=lstm_model)

# 调用fit函数训练模型
regressor.fit(x=train_X, y=train_y, batch_size=BATCH_SIZE, steps=TRAING_STEPS)

# 使用训练好的模型对测试数据进行预测
predicted = [[pred] for pred in regressor.predict(test_X)]

# 计算rmse作为评价指标
rmse = np.sqrt(((predicted - test_y) ** 2).mean(axis=0))
print 'Mean Square Error is: %f' % rmse[0]

# 对预测的sin函数曲线进行绘图
# 并存储到运行目录下sin,png
fig = plt.figure
plot_predicted = plt.plot(predicted, label='predicted')
plot_test = plt.plot(test_y, label='real_sin')
plt.legend([plot_predicted, plot_test], ['predicted', 'real_sin'])
fig.savefig('sin.pig')


















