#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 08:01:05 2017

@author: zach
"""

import tensorflow as tf

k = tf.placeholder(tf.float32)

# Make a normal distribution(正太分布), with a shifting mean
mean_moving_normal = tf.random_normal(shape=[1000], mean=(5*k), stddev=1)
# Record that distribution into a histogram summary
tf.summary.histogram("normal/moving_mean", mean_moving_normal)

# Make a normal distribution with shrinking variance(收缩的正太分布)
variance_shrinking_normal = tf.random_normal(shape=[1000], mean=0, stddev=1-(k))
# Record that distribution too
tf.summary.histogram("normal/shrinking_variance", variance_shrinking_normal)

# Let's combine both of those distributions into one dataset(连接标志正太分布和收缩的正太分布)
normal_combined = tf.concat([mean_moving_normal, variance_shrinking_normal], 0)
# We add another histogram summary to record the combined distribution
tf.summary.histogram("normal/bimodal", normal_combined)

# Add a gamma distribution（gamma 分布）
gamma = tf.random_gamma(shape=[1000], alpha=k)
tf.summary.histogram("gamma", gamma)

# And a poisson distribution（泊松分布）
poisson = tf.random_poisson(shape=[1000], lam=k)
tf.summary.histogram("poisson", poisson)

# And a uniform distribution(均值分布)
uniform = tf.random_uniform(shape=[1000], maxval=k*10)
tf.summary.histogram("uniform", uniform)

# Finally, combine everything together!(连接所有的分布)
all_distributions = [mean_moving_normal, variance_shrinking_normal,
                     gamma, poisson, uniform]
all_combined = tf.concat(all_distributions, 0)
tf.summary.histogram("all_combined", all_combined)

# 定义日志生成操作的总操作对象
# 执行一遍summaries相当于执行一遍所有的日志生成操作
summaries = tf.summary.merge_all()

# Setup a session and summary writer
sess = tf.Session()
writer = tf.summary.FileWriter("./histogram_example")

# Setup a loop and write the summaries to disk
N = 400
for step in range(N):
  k_val = step/float(N)
  summ = sess.run(summaries, feed_dict={k: k_val})
  writer.add_summary(summ, global_step=step)