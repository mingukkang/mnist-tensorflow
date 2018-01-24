import tensorflow as tf
import tensorflow.contrib.slim as slim
from function import *

def network(input,phase_train):
    X_img = tf.reshape(input, [-1,28,28,1], name ="X_img")
    with tf.variable_scope("conv1"):
        w1 = tf.get_variable("w1",initializer =  xavier_initializer([3,3,1,32],0.5,uniform =False))
        conv1 = tf.nn.conv2d(X_img,w1, strides = [1,1,1,1], padding ='SAME')
        conv1 = batch_normalization(conv1,phase_train)
        conv1 = tf.nn.relu(conv1)

    with tf.variable_scope("conv2"):
        w2 = tf.get_variable("w2",initializer = xavier_initializer([3, 3, 32, 128],0.5,uniform = False))
        conv2 = tf.nn.conv2d(conv1,w2,strides=[1,1,1,1],padding ='SAME')
        conv2 = batch_normalization(conv2,phase_train)
        conv2 = tf.nn.relu(conv2)

    with tf.variable_scope("max_pool1"):
        max_pool1 = slim.max_pool2d(conv2, [3,3], scope = "pooling1")
        max_pool1 = batch_normalization(max_pool1,phase_train)

    with tf.variable_scope("conv3"):
        w3 = tf.get_variable("w3",initializer = xavier_initializer([3, 3, 128, 256],0.5, uniform = False))
        conv3 = tf.nn.conv2d(max_pool1, w3,strides =[1,1,1,1], padding ='SAME')
        conv3 =  batch_normalization(conv3,phase_train)
        conv3 = tf.nn.relu(conv3)

    with tf.variable_scope("max_pool2"):
        max_pool2 = slim.max_pool2d(conv3, [3,3], scope ="pooling2")
        max_pool2 = batch_normalization(max_pool2,phase_train)

    with tf.variable_scope("flatten"):
        flatten_layer = slim.flatten(max_pool2)

    with tf.variable_scope("fully1"):
        w4 = tf.get_variable("w4",initializer = xavier_initializer([9216,1000], 0.5,uniform = False))
        b4 = tf.get_variable("b4", initializer =tf.random_normal([1000]))
        fully1 = tf.matmul(flatten_layer,w4) + b4
        fully1 = batch_normalization(fully1,phase_train)
        fully1 = tf.nn.relu(fully1)

    with tf.variable_scope("fully2"):
        w5 = tf.get_variable("w5",initializer = xavier_initializer([1000,10], 0.5, uniform = False))
        b5 = tf.get_variable("b5",initializer = tf.random_normal([10]))
        logits = tf.matmul(fully1,w5) +b5
        logits = batch_normalization(logits, phase_train)

    with tf.variable_scope("regularization"):
        w1_square_mean = tf.reduce_mean(tf.square(w1))
        w2_square_mean = tf.reduce_mean(tf.square(w2))
        w3_square_mean = tf.reduce_mean(tf.square(w3))
        w4_square_mean = tf.reduce_mean(tf.square(w4))
        w5_square_mean = tf.reduce_mean(tf.square(w5))
        l2 = w1_square_mean + w2_square_mean + w3_square_mean + w4_square_mean + w5_square_mean
    return logits, l2