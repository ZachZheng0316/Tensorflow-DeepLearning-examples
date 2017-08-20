#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 20 10:45:28 2017

@author: zach
"""

import tensorflow as tf

saver = tf.train.import_meta_graph("./addModel/addModel.ckpt.meta")

with tf.Session() as sess:
    saver.restore(sess, "./addModel/addModel.ckpt")
    print sess.run(tf.get_default_graph().get_tensor_by_name("add:0"))