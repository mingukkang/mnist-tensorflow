import tensorflow as tf

def Vgg_network():
    X_img = tf.placeholder(tf.float32, shape = [None,224,224,3], name = "Input_images")
    Y = tf.placeholder(tf.float32, shape = [None,1000], name = "Target")

    def  conv_block(input_data,num_block,num_conv,input_channel,output_channel):
        with tf.variable_scope(block + str(num_block)):
            for i in range(num_conv):
                j = tf.cast(i+1,tf.string)
                weights = tf.get_variable("weights"+ j ,shape = [3,3,num_input,num_output], initializer = tf.contirb.layers.xavier_initializer())
                conv = tf.nn.conv2d(input_data, weights, strides = [1,1,1,1], padding ="SAME")
                conv = tf.nn.relu(conv)
            max_pool = tf.nn.max_pool(conv, ksize = [1,2,2,1], stride = [1,2,2,1], padding = "SAME")
        return max_pool

    block1 = conv_block(X_img,1,2,3,64)
    block2 = conv_block(block1,2,2,64,128)
    block3 = conv_block(block2,3,4,128,256)
    block4 = conv_block(block3,4,4,256,512)
    block5 = conv_block(block4,5,4,512,512)
    flatten_layer = tf.contrib.layers.flatten(block5)
    fully_1 = tf.contrib.layers.fully_connected(flatten_layer,[7*7*512,4096], intializer = tf.contrib.layers.xavier_initializer(),name ="Fully_1")
    fully_2 = tf.contrib.layers.fully_connected(fully_1, [4096,4096], initializer = tf.contrib.layers.xavier_initializer(),name = "Fully_2")
    fully_3 = tf.contrib.layers.fully_connected(fully_2,[4096,1000], initializer = tf.contrib.layers.xavier_initializer(),name ="Fully_3")
    cost = tf.reduce_mean(tf.losses.softmax_cross_entropy(Y,fully_3))



