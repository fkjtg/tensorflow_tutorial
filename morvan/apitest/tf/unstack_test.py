import numpy as np
import tensorflow as tf

my_variable = tf.get_variable("my_variable", [1, 2, 3], tf.float32, initializer=tf.zeros_initializer)
# a = [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]
# a = np.array(a)
# print(a)
# b = tf.transpose(a)
# b = tf.unstack(b, axis=1)
initial = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(initial)
    b = tf.unstack(my_variable, axis=2)
    c = sess.run(b)
    print(c)

    my_variable = sess.run(my_variable)
    print(my_variable)
