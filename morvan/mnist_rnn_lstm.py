"""教程来源：https://morvanzhou.github.io/tutorials/machine-learning/tensorflow/5-08-RNN2/"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
lr = 0.001
training_iters = 100000
batch_size = 128

col_inputs = 28
row_inputs = 28
n_hidden_units = 128  # 隐藏层的神经元个数
n_classes = 10

x = tf.placeholder(tf.float32, [None, row_inputs, col_inputs])
y = tf.placeholder(tf.float32, [None, n_classes])

weight = {
    'in': tf.Variable(tf.random_normal([col_inputs, n_hidden_units])),
    'out': tf.Variable(tf.random_normal([n_hidden_units, n_classes]))
}

biases = {
    'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_units, ])),
    'out': tf.Variable(tf.constant(0.1, shape=[n_classes, ]))
}


def rnn(X, weights, biases):
    """
    RNN 主体结构, 这个 RNN 总共有 3 个组成部分 ( input_layer, cell, output_layer).
    :param X:
    :param weights:
    :param biases:
    :return:
    """

    # 首先，定义input_layer (数据转换见：https://blog.csdn.net/lanlana168/article/details/81136907)
    # X(128batch,28row,28col)
    # ==>X(128*28,28col)
    X = tf.reshape(X, [-1, col_inputs])
    # ==>X_in(128*28,28)
    X_in = tf.matmul(X, weights['in']) + biases['in']
    # ==>X_in(128,28,128)
    X_in = tf.reshape(X_in, [-1, row_inputs, n_hidden_units])

    # 接着，cell中计算。state划分成2部分（c_state,h_state）
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_units)
    init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)
    outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, X_in, initial_state=init_state, time_major=False)

    # 最后，output_layer和result的值
    # 方式一输出结果
    result = tf.matmul(final_state[1], weights['out'] + biases['out'])
    # 方式二输出结果
    # outputs = tf.unstack(tf.transpose(outputs, [1, 0, 2]))
    # result = tf.matmul(outputs[-1], weights['out'] + biases['out'])
    return result


pre = rnn(x, weight, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pre, labels=y))
train_op = tf.train.AdamOptimizer(lr).minimize(cost)

correct_pre = tf.equal(tf.argmax(pre, 1), tf.argmax(y, 1))
accuary = tf.reduce_mean(tf.cast(correct_pre, tf.float32))
with tf.Session() as sess:
    init = tf.global_variables_initializer()

    sess.run(init)

    step = 0

    while step * batch_size < training_iters:
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        batch_xs = batch_xs.reshape([batch_size, row_inputs, col_inputs])
        sess.run([train_op], feed_dict={x: batch_xs, y: batch_ys, })

        if step % 20 == 0:
            print(sess.run(accuary, feed_dict={x: batch_xs, y: batch_ys, }))
        step += 1
