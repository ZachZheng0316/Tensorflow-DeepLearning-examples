#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 20 10:51:41 2017

@author: zach
"""

import tensorflow as tf

s1 = tf.Variable(tf.constant(1.0, shape=[1]), name="other-v1")
s2 = tf.Variable(tf.constant(2.0, shape=[1]), name="other-v2")
result = s1 + s2

saver = tf.train.Saver({"v1": s1, "v2": s2})

with tf.Session() as sess:
    saver.restore(sess, "./addModel/addModel.ckpt")
    print sess.run(result)