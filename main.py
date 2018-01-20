import tensorflow as tf
import tensorflow.contrib.slim as slim
import function as f
import numpy as np

def network(input):
    X_img = tf.reshape(input, [-1,28,28,1], name ="X_img")
    with tf.variable_scope("conv1"):
        w1 = tf.Variable(f.xavier_initializer([3,3,1,32],0.5,uniform =False))
        conv1 = tf.nn.conv2d(X_img,w1, strides = [1,1,1,1], padding ='SAME')
        conv1 = f.batch_normalization(conv1)
        conv1 = tf.nn.relu(conv1)
    with tf.variable_scope("conv2"):
        w2 = tf.Variable(f.xavier_initializer([3, 3, 32, 128],0.5,uniform = False))
        conv2 = tf.nn.conv2d(conv1,w2,strides=[1,1,1,1],padding ='SAME')
        conv2 = f.batch_normalization(conv2)
        conv2 = tf.nn.relu(conv2)
    with tf.variable_scope("max_pool1"):
        max_pool1 = slim.max_pool2d(conv2, [3,3], scope = "pooling1")
    with tf.variable_scope("conv3"):
        w3 = tf.Variable(f.xavier_initializer([3, 3, 128, 256],0.5, uniform = False))
        conv3 = tf.nn.conv2d(max_pool1, w3,strides =[1,1,1,1], padding ='SAME')
        conv3 =  f.batch_normalization(conv3)
        conv3 = tf.nn.relu(conv3)
    with tf.variable_scope("max_pool2"):
        max_pool2 = slim.max_pool2d(conv3, [3,3], scope ="pooling2")
    with tf.variable_scope("flatten"):
        flatten_layer = slim.flatten(max_pool2)
    with tf.variable_scope("fully1"):
        w4 = tf.Variable(f.xavier_initializer([9216,1000], 0.5,uniform = False))
        fully1 = tf.matmul(flatten_layer,w4)
        fully1 = f.batch_normalization(fully1)
        fully1 = tf.nn.relu(fully1)
    with tf.variable_scope("fully2"):
        w5 = tf.Variable(f.xavier_initializer([1000,10], 0.5, uniform =False))
        b5 = tf.Variable(tf.random_normal([10]), name="b5")
        logits = tf.matmul(fully1,w5) +b5
    with tf.variable_scope("regularization"):
        w1_list = tf.reshape(w1,[-1])
        w2_list =tf.reshape(w2,[-1])
        w3_list = tf.reshape(w3,[-1])
        w4_list = tf.reshape(w4,[-1])
        w5_list = tf.reshape(w5,[-1])
        w_network = tf.concat([w1_list,w2_list,w3_list,w4_list,w5_list], axis = 0)
        w_square_sum = tf.reduce_sum(tf.square(w_network))
        n = tf.size(w_network,out_type = tf.float32)
        l2 = tf.divide(w_square_sum,n)
    return logits,l2