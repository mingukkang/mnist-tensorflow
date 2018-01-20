import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import main
import function

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
print(update_ops)
sess = tf.Session()
saver = tf.train.import_meta_graph('hyperpa15.ckpt.meta')
var_name = [v.name for v in tf.trainable_variables()]
var_dictionary = {v.name: v for v in tf.trainable_variables()}
sess.run(tf.initialize_all_variables())
print(sess, var_name)
weights = var_dictionary['fully1/bn/gamma:0']
want = sess.run(weights)

