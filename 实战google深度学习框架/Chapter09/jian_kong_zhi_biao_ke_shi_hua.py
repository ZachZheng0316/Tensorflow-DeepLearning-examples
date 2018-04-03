#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 19:44:13 2017

@author: zach
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

SUMMARY_DIR = "supervisor.log"
BATCH_SIZE = 1
TRAIN_STEPS = 3

# 生成Tensor var的监控项目和由var计算的标量的监控项目
def variable_summaries(var, name):
    with tf.name_scope('summaries'):
        # 生成Tensor var的监控项目
        tf.summary.histogram(name, var)
        
        # 计算var的平均值、生成平均值的监控项目
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean/' + name, mean)

        # 计算var的标准差、 生成标准差的监控项目
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev/' + name, stddev)

# 定义神经网络的前向传播过程
def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
    with tf.name_scope(layer_name):
        # 定义weights、并生成weight的监控项目、生成weight各元素平均值和标准差的监控项目
        with tf.name_scope('weights'):
            weights = tf.Variable(tf.truncated_normal([input_dim, output_dim], stddev=0.1))
            variable_summaries(weights, layer_name + '/weights')
        # 定义biases、并生成biases的监控项目、生成biases各个元素平均值和标准差的监控项目
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.constant(0.0, shape=[output_dim]))
            variable_summaries(biases, layer_name + '/biases')
        # 定义preactivate、并生成preactivate的监控项目、生成preactivate各个元素平均值和标准差的监控项目
        with tf.name_scope('Wx_plus_b'):
            preactivate = tf.matmul(input_tensor, weights) + biases
            tf.summary.histogram(layer_name + '/pre_activations', preactivate)
        # 计算preactivate的激活值
        activations = act(preactivate, name='activation')        
        
        # 记录神经网络节点输出在经过激活函数之后的分布。
        tf.summary.histogram(layer_name + '/activations', activations)
        return activations
    
def main():
    # 下载并加载MNIST_data，并one_hot化
    mnist = input_data.read_data_sets("./datasets/MNIST_data", one_hot=True)

    # 定义输入值x、y_
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, 784], name='x-input')
        y_ = tf.placeholder(tf.float32, [None, 10], name='y-input')

    # 生成image监控
    with tf.name_scope('input_reshape'):
        image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])
        tf.summary.image('input', image_shaped_input, 10)

    # 计算前向传播结果
    hidden1 = nn_layer(x, 784, 2, 'layer1')
    y = nn_layer(hidden1, 2, 10, 'layer2', act=tf.identity)

    # 计算交叉熵、并生成交叉熵的监控项目
    with tf.name_scope('cross_entropy'):
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))
        tf.summary.scalar('cross_entropy', cross_entropy)

    # 定义优化操作
    with tf.name_scope('train'):
        train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

    # 定义模型在当前给定的数据上的正确率，并定义生成正确率的监控项目。
    # 如果在sess.run时给定的数据是训练的batch，那么得到的准确率就是在这个训练batch上的正确率；
    # 如果给定的数据为验证或者测试数据，那么得到的正确率就是当前模型在验证或测试集数据上的正确率。
    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
            '''
                tf.equal(x, y, name=None)
                Args：
                    x,y: all are Tensor.
                returns：
                    Returns the truth value of (x == y) element-wise
            '''
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            '''
                tf.cast(x, dtype, name=None)
                Casts a tensor to a new type
                Args：
                    x： A Tensor or SparseTensor.
                    dtype: The destination type.
                    name: A name for the operation(optional)
                returns:
                    A Tensor or SparseTensor with sanme shape as x.
                For example:
                    # tensor 'a' is [1.8, 2.2], dtype=tf.float32
                    tf.cast(a, tf.int32) ==> [1, 2] # dtype=tf.int32
            '''
        tf.summary.scalar('accuracy', accuracy)

    # 和tf.summary.scalar、tf.summary.histogram和tf.summary.image类似，这些函数不会立即执行
    # 除非通过sess.run来明确调用这些函数。因为程序中定义的写日志操作比较多，一一调用很麻烦，所以
    # Tensorflow提供了tf.summary.merge_all()整理所有的日志生成操作。在Tensorflow中，只要运行
    # 这个操作就可以将代码中所有定义的日志生成操作执行一遍，从而将所有日志写入文件。
    merged = tf.summary.merge_all()

    with tf.Session() as sess:
        # 初始化写日志的writer,并将当前Tensorflow计算图写入日志
        summary_writer = tf.summary.FileWriter(SUMMARY_DIR, sess.graph)
        # 初始化所有的变量
        tf.global_variables_initializer().run()

        for i in range(TRAIN_STEPS):
            # 按BATCH_SIZE取出样本数据
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            # 运行训练步骤以及所有的日志生成操作，得到这次运行的日志。
            summary, _ = sess.run([merged, train_step], feed_dict={x: xs, y_: ys})
            # 将得到的所有日志写入日志文件，这样TensorBoard程序就可以拿到这次运行所对应的
            # 运行信息。
            summary_writer.add_summary(summary, i)

    summary_writer.close()
    
if __name__ == '__main__':
    main()