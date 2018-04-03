#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 20:23:35 2018

@author: zach
"""
# 加载库文件
import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_inference
import mnist_train

#每10s加载一次最新的模型
EVAL_INTERVAL_SECS = 10 #加载的时间间隔。

def evaluate(mnist):
    with tf.Graph().as_default() as g:
        #定义输入输出的格式和验证样本集
        x = tf.placeholder(tf.float32, [None, mnist_inference.INPUT_NODE], name='x-input')
        y_ = tf.placeholder(tf.float32, [None, mnist_inference.OUTPUT_NODE], name='y-input')
        validate_feed = {x: mnist.validation.images, y_: mnist.validation.labels}

        #计算前向传播结果、计算准确率、计算精度
        #因为测试结果不关注正则化损失函数的值，所以这里用于计算正则化损失函数被设置为None
        #使用tf.argmax(y, 1)就可以得到输入样例的预测类别
        y = mnist_inference.inference(x, None)
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        #构建滑动平均类
        #通过变量重命名的方式加载模型，这样在前向传播的过程中就不需要调用滑动平均类函数来获取平均值了
        #这样就可以完全使用mnist_inference.py中定义的前向传播的过程
        variable_averages = tf.train.ExponentialMovingAverage(mnist_train.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        #每隔EVAL_INTERVAL_SECS秒调用一次计算正确率的过程以检测训练过程中正确率的变化
        while True:
            with tf.Session() as sess:
                #tf.train.get_checkpoint_state函数会通过checkpoint文件自动找到目录中罪行的文件模型
                ckpt = tf.train.get_checkpoint_state(mnist_train.MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    #加载模型
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    #通过文件名得到模型保存时迭代的轮数
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    accuracy_score = sess.run(accuracy, feed_dict=validate_feed)
                    print("After %s training step(s), validation accuracy = %g" % (global_step, accuracy_score))
                else:
                    print('No checkpoint file found')
                    return
            time.sleep(EVAL_INTERVAL_SECS)

#主程序
def main(argv=None):
    mnist = input_data.read_data_sets("../../datasets/MNIST_data", one_hot=True)
    evaluate(mnist)

if __name__ == '__main__':
    main()
