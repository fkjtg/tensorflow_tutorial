import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
    return result


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, filter):
    """
    卷积技术

    x:提供卷积处理的数据

    filter=[patch_xs,patch_ys,in_size,out_size]

    strides=[1,x_movement,y_movement,1]
    strides的首尾必需都是1，
    第二位指x轴每一步移动多少个像素
    第三位指y轴每一步移动多少个像素

    padding有二种方式
    第一种是'VALID'
    第二种是'SAME'
    """
    return tf.nn.conv2d(x, filter, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    """
    池化技术
    起到的作用是处理压缩数据所以strides的第二位与第三位是2

    strides=[1,x_movement,y_movement,1]
    strides的首尾必需都是1，
    第二位指x轴每一步移动多少个像素
    第三位指y轴每一步移动多少个像素
    """
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# 定义卷积神经网络的占位
xs = tf.placeholder(tf.float32, [None, 784]) / 255.
ys = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)
x_image = tf.reshape(xs, [-1, 28, 28, 1])


def conv_pool_layer(w_shape, b_shape, x):
    """
    卷积池化
    先进行卷积过滤，再进行池化层压缩，保证数据完整性

    卷积过滤后的输出列与池化层的输入行需要保持一致
    w_shape=[patch_xs,patch_ys,in_size,out_size]

    """
    w_conv = weight_variable(w_shape)
    b_conv = bias_variable(b_shape)
    h_conv = tf.nn.relu(conv2d(x, w_conv) + b_conv)
    h_pool = max_pool_2x2(h_conv)
    return h_pool


conv_1 = conv_pool_layer([5, 5, 1, 32], [32], x_image)
conv_2 = conv_pool_layer([5, 5, 32, 64], [64], conv_1)

## fc1 layer （全连接层）##
w_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(conv_2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

## fc2 layer（全连接层） ##
w_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, w_fc2) + b_fc2)

##
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
init = tf.global_variables_initializer()
sess.run(init)

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 0.5})

    if i % 50 == 0:
        print(compute_accuracy(mnist.test.images[:1000], mnist.test.labels[:1000]))
