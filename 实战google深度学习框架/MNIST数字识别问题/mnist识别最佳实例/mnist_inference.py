#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 20:23:35 2018

@author: zach
"""

import tensorflow as tf

# 定义神经网络结构相关的参数
INPUT_NODE = 784
OUTPUT_NODE = 10
LAYER1_NODE = 500

# 定义获取每层参数的函数
# 通过tf.get_variable函数来获取变量。在训练神经网络时，会创建这些变量；在测试的时候，会通过保存的模型加载这些变量；
# 而且，更加方便的是，因为可以在变量加载时将滑动平均变量重命名，所以可以直接通过同样的名字在训练时使用变量，而在测试时
# 使用变量的滑动平均值。
# 在这个函数中也会将变量的正则化损失加入损失集合。
def get_weights_variable(shape, regularizer):
    weights = tf.get_variable("weights", shape=shape, initializer=tf.truncated_normal_initializer(stddev=0.1))

    # 当给出正则化函数时，将当前变量的正则化损失加入为loss的集合。
    # 在这里使用了add_to_collection函数将一个张量加入一个集合，而这个集合的名字为losses。
    # 这是自定义集合，不在TensorFlow自定义的集合之内
    if regularizer != None:
        tf.add_to_collection("losses", regularizer(weights))

    return weights

# 定义神经网络的前向传播过程
def inference(input_tensor, regularizer):
    # 申明第一层神经网络的变量，并完成前向传播的过程
    with tf.variable_scope('layer1', reuse=False):
        weights = get_weights_variable([INPUT_NODE, LAYER1_NODE], regularizer)
        biases = tf.get_variable("biases", shape=[LAYER1_NODE], initializer=tf.constant_initializer(0.0))
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)

    # 申明第二层神经网络的便利，并完成前向传播
    with tf.variable_scope('layer2', reuse=False):
        weights = get_weights_variable([LAYER1_NODE, OUTPUT_NODE], regularizer)
        biases = tf.get_variable("biases", shape=[OUTPUT_NODE], initializer=tf.constant_initializer(0.0))
        layer2 = tf.matmul(layer1, weights) + biases

    # 返回前向传播的结果
    return layer2
