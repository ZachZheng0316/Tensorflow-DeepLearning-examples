#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 20:23:35 2018

@author: zach
"""
import os

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 加载mnist_inference.py 中定义的常量和前向传播的函数
import mnist_inference

# 配置神经网络的参数
BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULAREZITION_RATE = 0.0001
TRAINING_STEPS = 30000
MOVING_AVERAGE_DECAY = 0.99

# 模型保存的路径和文件名
MODEL_SAVE_PATH = "./model/"
MODEL_NAME = "model"

# 定义训练过程
def train(mnist):

    # 定义输入输出的placeholder
    x = tf.placeholder(tf.float32, [None, mnist_inference.INPUT_NODE], name="x-input")
    y_ = tf.placeholder(tf.float32, [None, mnist_inference.OUTPUT_NODE], name="y-input")

    # 定义正则化类
    regularizer = tf.contrib.layers.l2_regularizer(REGULAREZITION_RATE)

    # 定义前向传播过程，并创建网络参数
    y = mnist_inference.inference(x, regularizer)

    # 申明指向迭代步数的变量
    global_step = tf.Variable(0, trainable=False)

    # 定义滑动平均类,并产生滑动操作
    variable_average = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variable_average_op = variable_average.apply(tf.trainable_variables())

    # 计算损失值：定义交叉熵，定义损失值
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(y_, 1), logits=y)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))

    # 定义学习率
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step, mnist.train.num_examples/BATCH_SIZE, LEARNING_RATE_DECAY)

    # 定义优化操作
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    with tf.control_dependencies([train_step, variable_average_op]):
        train_op = tf.no_op(name='train')

    # 初始化TensorFlow持久类
    saver = tf.train.Saver()

    with tf.Session() as sess:
        # 执行所有变量初始化操作
        tf.global_variables_initializer().run()

        # 训练模型：每个1000次保存一次模型
        for i in range(TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: xs, y_:ys})

            # 每1000次保存模型
            if i % 1000 == 0:
                # 输出当前训练情况。这里只输出了模型在当前batch上的损失函数大小。
                # 通过损失函数的大小可以大概了解训练的情况
                # 在验证数据集上的正确率会有一个单独的程序来完成
                print ("After %d training step(s), loss on training " "batch is %g." % (step, loss_value))

                # 保存当前模型
                # 注意这里给出了global_step参数，这样可以让每个被保存模型的文件名末尾加上训练的轮数
                # 比如:“model-1000”表示训练1000轮之后得到的模型
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)

def main(argv=None):
    mnist = input_data.read_data_sets("../../datasets/MNIST_data", one_hot=True)
    train(mnist)

if __name__ == '__main__':
    tf.app.run()
