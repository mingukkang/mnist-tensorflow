import tensorflow as tf

def  Resnet():
    X_img = tf.placeholder(tf.float32, shape = [None,224,224,3], name ="input_images")
    Y = tf.placeholder(tf.float32, shape  =[None,1000], name ="Target")

    w_start = tf.get_variable("w_start", shape = [7,7,3,64], initializer = tf.contrib.layers.xavier_initializer())
    conv_7x7_2 = tf.nn.conv2d(X_img,w_start,strides = [1,2,2,1], name ="start_conv",padding ="SAME")
    conv_7x7_2 = tf.nn.relu(conv_7x7_2)
    max_pool_1 = tf.nn.max_pool(conv_7x7_2,[1,3,3,1],[1,2,2,1],padding ="SAME")

    def residual_block(input_data,order,strides, output_channel):
        input_channel = tf.shape(input_data)[2]
        input_channel = tf.cast(input_channel,tf.int32)
        with variable_scope("block"+ str(order)):
            weights_1 = tf.get_variable("weights_1", shape = [1,1,input_channel,output_channel], initializer = tf.contrib.layers.xavier_initializer())
            conv = tf.nn.conv2d(input_data,weights_1, stride = [1,strides,strides,1], name = "conv1",padding ="SAME")
            conv = tf.nn.relu(conv)
            weights_2 = tf.get_variable("weights_2", shape =[3,3,output_channel,output_channel], initializer =tf.contrib.layers.xavier_initializer())
            conv = tf.nn.conv2d(conv,weights_2, stride =[1,1,1,1], name = "conv2",padding ="SAME")
            if strides ==1:
                conv = tf.concat([conv,input_data],3)
            conv = tf.nn.relu(conv)
            return conv

    block_1 = residual_block(max_pool_1,1,1,64)
    block_2 = residual_block(block_1,2,1,64)
    block_3 = residual_block(block_2,3,1,64)
    #_______________________________________#
    block_4 = residual_block(block_3,4,2,128)
    block_5 = residual_block(block_4,5,1,128)
    block_6 = residual_block(block_5,6,1,128)
    block_7 = residual_block(block_6,7,1,128)
    # _______________________________________#
    block_8 = residual_block(block_7,8,2,256)
    block_9 = residual_block(block_8,9,1,256)
    block_10 = residual_block(block_9, 10, 1, 256)
    block_11 = residual_block(block_10,11, 1, 256)
    block_12 = residual_block(block_11,12, 1, 256)
    block_13 = residual_block(block_12,13, 1, 256)
    # _______________________________________#
    block_14 = residual_block(block_13,14,2,512)
    block_15 = residual_block(block_14, 15, 1, 512)
    block_16 = residual_block(block_15, 16, 1, 512)
    Avg_pool = tf.nn.avg_pool(block_16,[1,7,7,1],stride =[1,1,1,1], padding ="VALID")
    flatten_layer = tf.reshape(Avg_pool, [-1,1024])
    fully_1 = tf.contrib.layers.fully_connected(flatten_layer,[1024,1000], initializer = tf.contrib.layers.xavier_initializer())
    cost = tf.reduce_mean(tf.nn.losses.softmax_cross_entropy(Y,fully_1))
