#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 07:58:30 2017

@author: zach
"""

import tensorflow as tf

x = [[1, 2], [3, 4]]  # shape = [2, 2]
y = [[5, 6], [7, 8]]  # shape = [2, 2]
z = [[9, 10], [11, 12]]  # shape = [2, 2]

t1 = tf.stack(values = [x, y, z], axis = 0) # ==>[[1, 4], 沿着axis=0的方向打包tensors, shape=[3, 2, 2]
                                            #     [2, 5], 
                                            #     [3, 6]]
t2 = tf.stack(values = [x, y, z], axis = 1) # ==>[[1, 2, 3], 沿着axis=1的方向打包tensors, shape=[2, 3, 2]
                                            #     [4, 5, 6]]
un1 = tf.unstack(value = t1, num = 3, axis = 0)
un2 = tf.unstack(value = t2, num = 3, axis = 1)

with tf.Session() as sess:
    print sess.run(t1)
    print sess.run(t2)
    print sess.run(un1)
    print sess.run(un2)