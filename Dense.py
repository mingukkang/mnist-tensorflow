import tensorflow as tf
from function import *

X_img = tf.placeholder(tf.float32,shape = [None,224,224,3], name ="input_images")
Y = tf.placeholder(tf.float32,shape = [None,1000], name ="Target")

weights = tf.get_variable("weights",shape =[7,7,3,64], initializer =tf.contrib.layers.xavier_initializer())
conv_7x7_2 = tf.nn.conv2d(X_img,weights,strides = [1,2,2,1],name ="7x7_2_conv",padding ="SAME")
max_pool_2 = tf.nn.max_pool(conv_7x7_2,[1,3,3,1],strides =[1,2,2,1],padding="SAME")

def dense(mode = None):
    def dense_element(input_data,bottle_channel,output_channel):
        holder.append(input_data)
        input_channel = tf.cast(tf.shape(input_data)[2],tf.int32)
        weights = tf.variable(shape = [1,1,input_channel, bottle_channel], initializer = tf.contrib.layers.xavier_initializer())
        x = tf.nn.conv2d(input_data,weights, strides =[1,1,1,1], padding ="SAME")
        x = batch_normalization(x,phase_train = mode)
        x = tf.nn.relu(x)
        weights = tf.variable(shape = [3,3,bottle_channel,output_channel], initializer =tf.contrib.layers.xavier_initializer())
        x = tf.nn.conv2d(x,weights,stride = [1,1,1,1],padding ="SAME")
        return x

    def dense_block(input_data,num_block,output_channel):
        holder = []
        holder.append(input_data)
        input_channel = tf.cast(tf.shape(input_data)[2], tf.int32)
        x = dense_element(X_img,input_channel,output_channel)
        x = batch_normalization(x,phase_train =mode)
        x = tf.nn.relu(x)
        holder.append(x)
        for i in range(num_block- 3):
            x = dense_element(x,input_channel,output_channel)
            for j in range(i+1):
                temp = holder[j]
                x = tf.concat([x,temp],3)
            x = batch_normalization(x,phase_train = mode)
            x = tf.nn.relu(x)
            holder.append(x) ## input
            x = dense_element(x,input_channel,output_channel)
        return x

    dense_block1= dence_block(max_pool_2,6,64)
    weights = tf.variable(shape = [1,1,64,64], initializer = tf.contrib.layers.xavier_initializer())
    transition_1 =  tf.nn.conv2d(dense_block1,weights,stride =[1,1,1,1],padding ="True")
    transition_1 = tf.nn.pool(transition_1,[1,2,2,1],"AVG",[1,2,2,1], padding ="VALID")
    dense_block2 = dence_block(transition_1,12,128)
    weights_2 = tf.variable(shape = [1,1,128,128],initializer =tf.contrib.layers.xavier_initializer())
    transition_2 = tf.nn.conv2d(dense_block2,weights_2,stride = [1,1,1,1],padding ="True")
    transition_2 = tf.nn.pool(transition_2,[1,2,2,1],"AVG",[1,2,2,1], padding ="VALID")
    dense_block3 = dence_block(transition_2,24,256)
    weights_3 = tf.variable(shape = [1,1,256,256],initializer =tf.contrib.layers.xavier_initializer())
    transition_3 = tf.nn.conv2d(dense_block3,weights_3,stride = [1,1,1,1],padding ="True")
    transition_3 = tf.nn.pool(transition_3,[1,2,2,1],"AVG",[1,2,2,1], padding ="VALID")
    dense_block4 = dence_block(transition_3,16,512)
    weights_4 = tf.variable(shape = [1,1,512,512],initializer =tf.contrib.layers.xavier_initializer())
    transition_4 = tf.nn.conv2d(dense_block4,weights_4,stride = [1,1,1,1],padding ="True")
    Avg_pool = tf.nn.avg_pool(transition_4, [1, 7, 7, 1], stride=[1, 1, 1, 1], padding="VALID")
    flatten = tf.contrib.layers.flatten(Avg_pool)
    fully_1 = tf.contrib.layers.fully_connected(flatten,[5*512,1000],initializer = tf.contrib.layers.xavier_initializer())
    fully_1 = batch_normalization(fully_1,phase_train = mode)
    cost = tf.reduce_mean(tf.nn.losses.softmax.cross_entropy(Y,fully_1))