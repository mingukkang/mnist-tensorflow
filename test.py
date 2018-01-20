import tensorflow as tf
import tensorflow.contrib.slim as slim
import main
import function
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot =True)
test_xs = mnist.test.images
test_ys =mnist.test.labels
total_batch_test = int(test_xs.shape[0]/500)

def test():
    X = tf.placeholder(tf.float32, shape = [None,784], name = "Test_input")
    Y = tf.placeholder(tf.float32, shape =[None,10], name ="Test_target")
    hypothesis_true,l2 = main.network(X)
    is_correct_test = tf.equal(tf.argmax(hypothesis_true,1), tf.argmax(Y,1))
    Accur_test = tf.reduce_mean(tf.cast(is_correct_test,dtype = tf.float32))

    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

    saver = tf.train.Saver()
    saver.restore(sess, 'hyperpa15.ckpt')
    # saver = tf.train.import_meta_graph('hyperpa15.ckpt.meta')
    # var_name = [v.name for v in  tf.trainable_variables()]
    # var_dictionary = {v.name: v for v in tf.trainable_variables()}
    # print(sess,var_name)
    Accuracy_test = 0
    for k in range(total_batch_test):
        batch_xs_test ,batch_ys_test = function.make_batch(test_xs,test_ys,500,k)
        feed_dict = {X:batch_xs_test, Y:batch_ys_test}
        aa = sess.run([Accur_test], feed_dict = feed_dict)
        Accuracy_test += aa[0]/total_batch_test
    print("Accuracy_test: ", Accuracy_test)

if __name__ =='__main__':
    # flags = tf.app.flags
    # FLAGS = flags.FLAGS
    # flags.DEFINE_integer("batch_size_test",500,"batch size for test")
    test()
