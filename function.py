import math
import tensorflow as tf
import numpy as np

function_list = ['make_batch','xavier_initializer','batch_normalization','flatten']
# make_batch(xs,ys,batch_size,step) # step 은 몇번재 배치인지를 나타내는 배치의 스텝임 ex) j
# xavier_initializer(shape, factor,uniform) #
#
#  factor은 He initializer를 하기위한 것으로 1/2가 되면 He initializer가 된다.
# batch_normalization(x) # convolution or fc연산 후에 나오는 값을 넣어주면 된다.
# flatten(input_data) # rank가 2 혹은 4인 데이터를 넣어주면 된다.
##########################################################

def make_batch(xs, ys, batch_size, step):
    length = len(xs)
    total_batch = int(length/batch_size)
    if step != (total_batch):
        batch_xs = xs[step*batch_size:step*batch_size+batch_size]
        batch_ys = ys[step*batch_size:step*batch_size+batch_size]
    else:
        batch_xs = xs[step*batch_size:]
        batch_ys = xs[step*batch_size:]
    return batch_xs, batch_ys

# 조금 이상있음 ㅠㅠ#########################################################
def xavier_initializer(shape, factor, uniform = None):
    factor = float(factor)
    if len(shape) == 2:
        fan_in = 1
        fan_out = 1
    else: # len(shape) = 4
        fan_in = shape[-2]
        fan_out = shape[-1]
    for dim in shape[0:-2]:
        fan_in *=dim
        fan_out*=dim

    if uniform:
        lim =  math.sqrt((3.0/n)*factor)
        result = tf.random_uniform(shape,-1*lim, lim,dtype = tf.float32)
    else:
        variation = (2/( fan_in +fan_out))*factor
        dev = math.sqrt(variation)
        result = tf.truncated_normal(shape,mean = 0, stddev = dev)
    return result

##########################################################
def normalize(x, mean, var, beta, gamma):
    inv = ((var + 0.001) ** (-0.5))
    inv = inv * gamma
    return (x - mean) * inv + beta

def batch_normalization(x,scope_name = "bn"):
    input_shape = x.shape
    if len(input_shape) == 4:
        n_out = input_shape[-1]
        moment_shape = [0,1,2]
    elif len(input_shape) ==2:
        n_out = 1
        moment_shape = [0]
    with tf.variable_scope(scope_name):
        beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
                           name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1, shape=[n_out]),
                            name='gamma', trainable=True)
        batch_mean, batch_var = tf.nn.moments(x, moment_shape, name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.99)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = mean_var_with_update()
        return normalize(x, mean, var, beta, gamma)

###########################################################
def flatten(input_data):
    input_shape = input_data.shape

    def sec_array_flatten(input_matrix):
        saver = np.array([])
        for i in range(input_matrix.shape[-2]):
            saver = np.concatenate((saver ,input_matrix[i,:]), axis = 0)
        return saver

    def image_flatten(input_matrix):
        for j in range(input_matrix.shape[-4]):
            for k in range(input_matrix.shape[-1]):
                temp_holder1 = np.array(input_matrix[j, : , : ,k])
                saver  = sec_array_flatten(temp_holder1)
        return saver

    if len(input_shape) == 2:
        result = sec_array_flatten(input_matrix = input_data)
    elif len(input_shape) == 4:
        result = image_flatten(input_matrix = input_data)
    else:
        result = print("Error Account! plz this function activate when rank of matirx is 2 or 4")
    return result

def shuffle_data(twod_images, labels):
    num_images = twod_images.shape[0]
    idx = np.random.permutation(num_images)
    train_ig = twod_images[idx]
    train_lb = labels[idx]
    print("shuffling...")
    return train_ig, train_lb
