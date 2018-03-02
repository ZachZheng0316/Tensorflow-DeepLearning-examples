# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 07:02:55 2018

@author: wxyz
"""

# 导入库文件
import tensorflow as tf
import numpy as np

# 产生例子数据
M = np.random.randint(-10, 10, size=[5, 5, 3])
# 调整数据格式
M = np.asarray(M, dtype='float32')
M = M.reshape(1, 5, 5, 3)

# 定义卷积过滤器，深度为6
filter_weight = tf.get_variable('weight',
                                shape=[5, 5, 3, 6],
                                initializer=tf.truncated_normal_initializer(stddev=0.1))
# 设置偏置项
biases = tf.get_variable('biase', shape=[6], initializer=tf.constant_initializer(0.1))

# 计算卷积过程
# 构建输入节点
x = tf.placeholder('float32', [1, None, None, None])

# 构建卷积层
conv = tf.nn.conv2d(x, filter_weight, strides=[1, 2, 2, 1], padding='VALID')
layer = tf.nn.bias_add(conv, biases)

#构建池化层
pool = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

# 建立会话
with tf.Session() as sess:
    # 变量初始化
    tf.global_variables_initializer().run()
    
    # 计算卷积层
    convluted_M = sess.run(layer, feed_dict={x:M})
    
    # 计算池化层
    pooled_M = sess.run(pool, feed_dict={x:M})
    
    # 打印结果
    print("convoluted_M.shape: \n", convluted_M.shape)
    print("convoluted_M: \n", convluted_M)
    print("pooled_M: \n", pooled_M.shape)
    print("pooled_M: \n", pooled_M)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    