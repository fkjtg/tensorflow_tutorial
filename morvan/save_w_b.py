import tensorflow as tf
import numpy as np


def save_w_b():
    w = tf.Variable([[1, 2, 3], [3, 4, 5]], dtype=tf.float32, name='weights')
    b = tf.Variable([[1, 2, 3]], dtype=tf.float32, name='biases')

    initial = tf.global_variables_initializer()
    save = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(initial)
        save_path = save.save(sess, "my_w_b/w_b.ckpt")


def restore():
    w = tf.Variable(np.arange(6).reshape(2, 3), dtype=tf.float32, name='weights')
    b = tf.Variable(np.arange(3).reshape(1, 3), dtype=tf.float32, name='biases')

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, 'my_w_b/w_b.ckpt')
        print('weights', sess.run(w), 'biases', sess.run(b))


# save_w_b()
restore()
