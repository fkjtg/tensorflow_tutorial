import tensorflow as tf

import tensorflow as tf

with tf.name_scope('input1'):
    input1 = tf.constant([1.0, 2.0, 3.0], name="input1")
with tf.name_scope('input2'):
    input2 = tf.Variable(tf.random_uniform([3]), name="input2")
output = tf.add_n([input1, input2], name="add")
writer = tf.summary.FileWriter("logs/", tf.get_default_graph())
writer.close()

# import tensorflow as tf
#
# a = tf.constant([1.0, 2.0, 3.0], name='input1')
# b = tf.Variable(tf.random_uniform([3]), name='input2')
# add = tf.add_n([a, b], name='addOP')
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     writer = tf.summary.FileWriter("logs/", sess.graph)
#     print(sess.run(add))
# writer.close()
