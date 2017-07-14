#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 07:58:30 2017

@author: zach
"""

import tensorflow as tf

t1 = tf.constant([[1, 2, 3], [4, 5, 6]])
t2 = tf.constant([[7, 8, 9], [10, 11, 12]])

c1 = tf.concat(values=[t1, t2], axis=0, name="concat") # cat on rank-0
c2 = tf.concat(values=[t1, t2], axis=1, name="concat") # cat om rank-1

r1 = tf.reshape(tensor=c1, shape=[- 1], name=None)
r2 = tf.reshape(tensor=c2, shape=[- 1], name=None)

p1 = tf.stack(values=[t1, t2], axis=0, name="stack")
p2 = tf.stack(values=[t1, t2], axis=1, name="stack")
p3 = [t1, t2]

with tf.Session() as sess:
    print 't1: '
    print sess.run(fetches=t1, feed_dict=None, options=None, run_metadata=None)
    
    print 't2: '
    print sess.run(fetches=t2, feed_dict=None, options=None, run_metadata=None)
    
    print 'c1: '
    print sess.run(fetches=c1, feed_dict=None, options=None, run_metadata=None)
    
    print 'c2: '
    print sess.run(fetches=c2, feed_dict=None, options=None, run_metadata=None)
    
    print '[c1, c2]: '
    print sess.run(fetches=[c1, c2],feed_dict=None, options=None, run_metadata=None)
    
    print 'rank(t1), rank(c1), rank(r1), rank(p1)'
    print sess.run(fetches=[tf.rank(input=t1, name=None), 
                            tf.rank(input=c1, name=None), 
                            tf.rank(input=r1, name=None), 
                            tf.rank(input=p1, name=None)], 
                    feed_dict=None, options=None, run_metadata=None)
    
    print 'r1: '
    print sess.run(fetches=r1, feed_dict=None, options=None, run_metadata=None)
    
    print 'r2: '
    print sess.run(fetches=r2, feed_dict=None, options=None, run_metadata=None)
    
    print 'shape(p1), shape(p2): '