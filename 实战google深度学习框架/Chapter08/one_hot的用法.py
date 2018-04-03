#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 07:59:34 2017

@author: zach
"""

import tensorflow as tf

#indices = [1, 0, -1, 2]
#indices = [[0, 2], [1, -1]]
indices = [1, 2, 3]
depth = 3
on_value = 1
off_value = 0
axis = 1

#axis = -1
target = tf.one_hot(indices, depth=depth, on_value=on_value, off_value=off_value, axis=axis)


with tf.Session() as sess:
    print sess.run(target)