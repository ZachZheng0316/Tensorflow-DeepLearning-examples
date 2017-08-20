#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 20 11:59:52 2017

@author: zach
"""

import tensorflow as tf

v1 = tf.Variable(tf.constant(1.0, shape=[1]), name="v1")
v2 = tf.Variable(tf.constant(2.0, shape=[1]), name="v2")
result = v1 + v2

init_op = tf.variables_initializer([v2])

saver = tf.train.Saver([v1])

with tf.Session() as sess:
    sess.run(init_op)
    saver.restore(sess, "subModel/subModel.ckpt")
    print sess.run(result)