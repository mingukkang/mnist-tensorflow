import tensorflow as tf
import tensorflow.contrib.slim as slim
import  main
import function
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot =True)
train_xs = mnist.train.images[0:50000]
train_ys = mnist.train.labels[0:50000]
val_xs = mnist.train.images[50000:]
val_ys = mnist.train.labels[50000:]

train_length = len(train_xs)
val_length = len(val_xs)

def train(mode = None):
    X = tf.placeholder(tf.float32, shape =[None,784], name = "Input_data")
    Y = tf.placeholder(tf.float32, shape = [None,10], name = "Target")
    global_step = tf.Variable(0, trainable =False)
    hypothesis ,l2= main.network(X)
    cost = tf.reduce_mean(tf.losses.softmax_cross_entropy(Y,hypothesis)) + FLAGS.lambda_val* l2
    learning_rate_decayed= FLAGS.learning_rate * FLAGS.decay_rate** (global_step / FLAGS.decay_step)
    optimizer = tf.train.AdamOptimizer(learning_rate_decayed).minimize(cost,global_step = global_step)
    is_correct = tf.equal(tf.argmax(hypothesis,1), tf.argmax(Y,1))
    Accuracy = tf.reduce_mean(tf.cast(is_correct,dtype = tf.float32))
    Learning_rate_summary = tf.summary.scalar("Learning_rate", learning_rate_decayed)
    Cost_summary = tf.summary.scalar("Cost",cost)
    Accur_summary = tf.summary.scalar("Accuracy",Accuracy)
    summary = tf.summary.merge([Cost_summary,Accur_summary,Learning_rate_summary])

    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)

    saver = tf.train.Saver()
    # tensorboard --inspect --logdir=./  ## 여기서 띄어쓰기가 아주 중요하니 조심하자!
    # 가상환경 확인은 conda env list, 나의 가상환경은 tensorflow_envs이다.
    writer =tf.summary.FileWriter(FLAGS.tensorboard_dir)
    writer.add_graph(sess.graph)

    for i in range(FLAGS.num_epoch):
        cost_val = 0
        Accur_train = 0
        Accur_validation = 0
        total_batch = int(train_length/FLAGS.batch_size)
        for j in range(total_batch):
            batch_xs, batch_ys =function.make_batch(train_xs,train_ys,FLAGS.batch_size, j)
            feed_dict = {X:batch_xs, Y:batch_ys}
            c,_,s,a = sess.run([cost,optimizer,summary,Accuracy], feed_dict = feed_dict)
            cost_val +=c/total_batch
            Accur_train +=a/total_batch
            writer.add_summary(s,i*total_batch + j)
        if i %5 ==0:
            location = "./hyperpa" + str(i) + ".ckpt"
            saving_data =saver.save(sess,location)

        if i % 1 ==0:
            Accur_validation = sess.run(Accuracy, feed_dict ={X:val_xs, Y:val_ys})
            print("Epoch: %d\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tCost_val : %06f \nTraining_Accuracy: %06f \t\t\t\t\tValidation_Accuracy: %06f"
                       % (i+1,cost_val,Accur_train,Accur_validation))
    sess.close()

if __name__ =='__main__':
    flags = tf.app.flags
    FLAGS = flags.FLAGS
    flags.DEFINE_float('lambda_val', 0.9,'lambda for regularization')
    flags.DEFINE_float('learning_rate', 0.001, 'learning rate for training')
    flags.DEFINE_float('decay_rate',0.95,'learning rate decay rate')
    flags.DEFINE_integer('decay_step',100,'decay step for learning rate decay')
    flags.DEFINE_integer('batch_size', 64, 'batch size for training')
    flags.DEFINE_integer('num_epoch', 20, 'number of epoch')
    flags.DEFINE_string('tensorboard_dir', './logs/data', 'log dir for tensorboard')
    train()

