import tensorflow as tf
import numpy as np

log_path = "logs/"

inputX = np.random.rand(3000, 1)
noise = np.random.normal(0, 0.05, inputX.shape)
outputY = inputX*4+1+noise

with tf.name_scope('inputs'):
    x1 = tf.placeholder(tf.float64, [None, 1], name='x1')
    y = tf.placeholder(tf.float64, [None, 1], name='y')



weight1 = tf.Variable(np.random.rand(inputX.shape[1], 4), name='weight1')
bias1 = tf.Variable(np.random.rand(inputX.shape[1], 4), name='bias1')
tf.summary.histogram('weight_1', weight1)


y1_ = tf.matmul(x1, weight1) + bias1

with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(y1_ - y), reduction_indices=[1]), name='loss')
    tf.summary.scalar('loss', loss)
with tf.name_scope('train'):
    train = tf.train.GradientDescentOptimizer(0.25).minimize(loss)

init = tf.initialize_all_variables()

with tf.Session() as sess:
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter(log_path, sess.graph)
    sess.run(init)
    for i in range(1000):
        sess.run(train, feed_dict={x1: inputX, y: outputY})
        summary = sess.run(merged, feed_dict={x1: inputX, y: outputY})
        writer.add_summary(summary, i)  # Write summary
        if i%50 == 0:
            writer.flush()
    writer.close()
    print(weight1.eval(sess))
    print('-------------------------------------')
    print(bias1.eval(sess))








