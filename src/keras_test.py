import tensorflow as tf
import numpy as np
weight1 = tf.constant([1,2,3,4])
x1 = tf.placeholder(tf.int32, [None, 1])
output = weight1*x1

with tf.Session() as sess:
    print(sess.run(output, feed_dict={x1: [[1],[2],[3]]}))

