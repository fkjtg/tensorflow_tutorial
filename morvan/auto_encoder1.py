import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from tensorflow.examples.tutorials.mnist import input_data
from mpl_toolkits.mplot3d import Axes3D

batch_size = 64
lr = 0.002
n_test_img = 5

mnist = input_data.read_data_sets('MNIST_data', one_hot=False)
test_x = mnist.test.images[:200]
test_y = mnist.test.labels[:200]

print(mnist.train.images.shape)
print(mnist.train.labels.shape)
plt.imshow(mnist.train.images[0].reshape((28, 28)), cmap='gray')
plt.title('%i' % np.argmax(mnist.train.labels[0]))
plt.show()

tf_x = tf.placeholder(tf.float32, [None, 28 * 28])

# encoder
e0 = tf.layers.dense(tf_x, 128, tf.nn.tanh)
e1 = tf.layers.dense(e0, 64, tf.nn.tanh)
e2 = tf.layers.dense(e1, 12, tf.nn.tanh)
encode = tf.layers.dense(e2, 3)

# decode
d0 = tf.layers.dense(encode, 12, tf.nn.tanh)
d1 = tf.layers.dense(d0, 64, tf.nn.tanh)
d2 = tf.layers.dense(d1, 128, tf.nn.tanh)
decode = tf.layers.dense(d2, 28 * 28, tf.nn.sigmoid)

loss = tf.losses.mean_squared_error(labels=tf_x, predictions=decode)
train = tf.train.AdamOptimizer(lr).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
f, a = plt.subplots(2, n_test_img, figsize=(5, 2))
plt.ion()

view_data = mnist.test.images[:n_test_img]

for i in range(n_test_img):
    a[0][i].imshow(np.reshape(view_data[i], (28, 28)), cmap='gray')
    a[0][i].set_xticks(())
    a[0][i].set_yticks(())

for step in range(8000):
    b_x, b_y = mnist.train.next_batch(batch_size)
    _, encode_, decode_, loss_ = sess.run([train, encode, decode, loss], {tf_x: b_x})

    if step % 100 == 0:
        print('train loss:%.4f' % loss_)
        decode_data = sess.run(decode, {tf_x: view_data})
        for i in range(n_test_img):
            a[1][i].clear()
            a[1][i].imshow(np.reshape(decode_data[i], (28, 28)), cmap='gray')
            a[1][i].set_xticks(())
            a[1][i].set_yticks(())
        plt.draw()
        plt.pause(0.01)
plt.ioff()

view_data = test_x[:200]
encoded_data = sess.run(encode, {tf_x: view_data})
fig = plt.figure(2)
ax = Axes3D(fig)
X, Y, Z = encoded_data[:, 0], encoded_data[:, 1], encoded_data[:, 2]
for x, y, z, s in zip(X, Y, Z, test_y):
    c = cm.rainbow(int(255 * s / 9))
    ax.text(x, y, z, s, backgroudcolor=c)
ax.set_xlim(X.min(), X.max())
ax.set_ylim(Y.min(), Y.max())
ax.set_zlim(Z.min(), Z.max())
plt.show()
