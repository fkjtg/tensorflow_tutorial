import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
lr = 0.001
training_iters = 100000
batch_size = 128

n_inputs = 28
n_steps = 28
n_hidden_units = 128
n_classes = 10

x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_classes])

weight = {
    'in': tf.Variable(tf.random_normal([n_inputs, n_hidden_units])),
    'out': tf.Variable(tf.random_normal([n_hidden_units, n_classes]))
}

biases = {
    'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_units, ])),
    'out': tf.Variable(tf.constant(0.1, shape=[n_classes, ]))
}


def rnn(x, weights, biases):
    X = tf.reshape(x, [-1, n_inputs])

    X_in = tf.matmul(X, weights['in']) + biases['in']
    X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_units])

    # if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
    #     cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True)
    # else:
    #     cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_units)
    cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_units)

    init_state = cell.zero_state(batch_size, dtype=tf.float32)
    outputs, final_state = tf.nn.dynamic_rnn(cell, X_in, initial_state=init_state, time_major=False)
    outputs = tf.unstack(tf.transpose(outputs, [1, 0, 2]))
    result = tf.matmul(outputs[-1], weights['out'] + biases['out'])
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
        batch_xs = batch_xs.reshape([batch_size, n_steps, n_inputs])
        sess.run([train_op], feed_dict={x: batch_xs, y: batch_ys, })

        if step % 20 == 0:
            print(sess.run(accuary, feed_dict={x: batch_xs, y: batch_ys, }))
        step += 1
