
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import LeNet5_infernece
import os
import numpy as np


# #### 1. 定义神经网络相关的参数

# In[2]:


# 配置神经网络训练参数
BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.01
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 30000
MOVING_AVERAGE_DECAY = 0.99

# 申明模型保存的路径
MODEL_SAVE_PATH = "./model/"
MODEL_NAME = "LeNet-5"


# #### 2. 定义训练过程

# In[3]:


def train(mnist):
    # 定义输出为4维矩阵的placeholder
    x = tf.placeholder(tf.float32, [
            BATCH_SIZE,                    # 第一维表示一个batch中样例的个数
            LeNet5_infernece.IMAGE_SIZE,   # 第二维和第三维表示图片的尺寸
            LeNet5_infernece.IMAGE_SIZE,
            LeNet5_infernece.NUM_CHANNELS],# 第四维表示图片的深度，对于RBG格式的图片，深度为5，为什么是5
        name='x-input')
    y_ = tf.placeholder(tf.float32, [None, LeNet5_infernece.OUTPUT_NODE], name='y-input')
    
    # 定义正则化类
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    
    # 构建前向传播计算图
    y = LeNet5_infernece.inference(x, True, regularizer)
    
    # 定义迭代变量
    global_step = tf.Variable(0, trainable=False)

    # 定义损失函数、学习率、滑动平均操作以及训练过程。
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        mnist.train.num_examples / BATCH_SIZE, 
        LEARNING_RATE_DECAY,
        staircase=True)

    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    with tf.control_dependencies([train_step, variables_averages_op]):
        train_op = tf.no_op(name='train')
        
    # 初始化TensorFlow持久化类。
    saver = tf.train.Saver()
    
    # 在会话中运行计算图
    with tf.Session() as sess:
        # 执行变量初始化操作
        tf.global_variables_initializer().run()
        
        # 迭代优化
        for i in range(TRAINING_STEPS):
            # 获取MNist数据
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            
            # 将训练的输入的训练数据格式调整为一个四维矩阵，并将这个调整后的数据传入sess.run的过程
            reshaped_xs = np.reshape(xs, (
                BATCH_SIZE,
                LeNet5_infernece.IMAGE_SIZE,
                LeNet5_infernece.IMAGE_SIZE,
                LeNet5_infernece.NUM_CHANNELS))
            
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: reshaped_xs, y_: ys})

            if i % 1000 == 0:
                # 输出当前训练情况。这里只输出了模型在当前batch上的损失函数大小
                print("After %d training step(s), loss on training batch is %g." % (step, loss_value))
                
                # 保存当前模型
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)


# #### 3. 主程序入口

# In[4]:


def main(argv=None):
    mnist = input_data.read_data_sets("../datasets/MNIST_data", one_hot=True)
    train(mnist)

if __name__ == '__main__':
    main()

