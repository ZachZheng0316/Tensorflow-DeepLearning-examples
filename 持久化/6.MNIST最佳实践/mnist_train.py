
# coding: utf-8

# #### 1.加载库文件

# In[1]:


import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_inference


# #### 2.定义神经网络结果相关参数

# In[2]:


BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULARAZTION_RATE = 0.0001
TRAINING_STEPS = 30000
MOVING_AVERAGE_DECAY = 0.99

# 模型保存的路径和文件名
MODEL_SAVE_PATH = "model/"
MODEL_NAME = "model.ckpt"


# #### 3.定义训练过程

# In[3]:


def train(mnist):
    # 定义输入输出placement
    x = tf.placeholder(tf.float32, [None, mnist_inference.INPUT_NODE], name="x-input")
    y_ = tf.placeholder(tf.float32, [None, mnist_inference.OUTPUT_NODE], name="y-input")
    
    # 定义正则化类
    regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)
    
    # 定义前向传播的过程
    y = mnist_inference.inference(x, regularizer)
    
    # 定义步数变量
    global_step = tf.Variable(0, trainable=False)
    
    # 定义滑动平均类及其操作
    variable_averages = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY, global_step)
    variable_averages_op = variable_averages.apply(
        tf.trainable_variables())
    
    # 定义交叉熵
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection("losses"))
    
    # 定义学习率
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        mnist.train.num_examples / BATCH_SIZE,
        LEARNING_RATE_DECAY)
    
    # 定义优化操作
    train_step = tf.train.GradientDescentOptimizer(learning_rate)        .minimize(loss, global_step=global_step)
        
    # 定义依赖操作
    with tf.control_dependencies([train_step, variable_averages_op]):
        train_op = tf.no_op(name="train")
        
    # 初始化持久化TensorFlow持久化类
    saver = tf.train.Saver(max_to_keep=30)
    
    # 定义会话
    config = tf.ConfigProto(allow_soft_placement= True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        # 变量初始化
        tf.global_variables_initializer().run()

        # 在训练过程中不再测试模型在验证数据上的表现，验证和测试的过程将会有一个独立
        # 的程序来完成
        for i in range(TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            _, loss_value, step = sess.run([train_op, loss, global_step],
                                          feed_dict={x: xs, y_:ys})
            
            # 每1000轮保存一次模型
            if i % 1000 == 0:
                # 输出当前训练情况。
                # 这里只输出了模型在当前训练batch上的损失函数大小
                # 通过损失函数的大小可以大概了解训练的情况。
                # 在验证数据集上的正确率会有一个单独的程序来生成
                print("Afetr %d training step(s), loss on training "
                     "batch is %g." % (step, loss_value))
                
                # 保存当前的模型
                # 注意这里给出了global_step参数，这样可以让美俄被保存的模型的文件名
                # 末尾加上训练的轮数，比如“model.ckpt-1000”表示训练1000轮之后得到
                # 的模型
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME),
                              global_step=global_step)


# #### 4.主程序入口

# In[4]:


def main(argv=None):
    tf.reset_default_graph()
    mnist = input_data.read_data_sets("../../../../datasets/MNIST_data/", one_hot=True)
    train(mnist)
    
if __name__=='__main__':
    tf.app.run()


# In[5]:


get_ipython().magic('tb')


# In[ ]:




