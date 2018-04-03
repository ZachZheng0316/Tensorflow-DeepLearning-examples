#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 20 14:13:37 2017

@author: zach
"""
import sys
import tensorflow as tf

stdi, stdo, stde = sys.stdin, sys.stdout, sys.stderr
reload(sys)
sys.stdin, sys.stdout, sys.stderr = stdi, stdo, stde
sys.setdefaultencoding('utf-8')



v = tf.Variable(0, dtype=tf.float32, name="v")

for variable in tf.global_variables():
    print variable.name
    
ema = tf.train.ExponentialMovingAverage(0.99)
maintain_averages_op = ema.apply(tf.global_variables())

for variables in tf.global_variables():
    print variables.name
    
saver = tf.train.Saver()
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    
    sess.run(tf.assign(v, 10))
    sess.run(maintain_averages_op)
    
    saver.save(sess, "./movingAveragesMode/movingAveragesMode.ckpt")
    print sess.run([v, ema.average(v)])