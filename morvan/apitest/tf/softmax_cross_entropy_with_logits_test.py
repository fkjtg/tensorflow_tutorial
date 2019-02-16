import tensorflow as tf

logits = tf.constant([[1., 2., 3.], [1., 2., 3.], [1., 2., 3.]])
y = tf.nn.softmax(logits)

y_ = tf.constant([[0., 0., 1.], [0., 0., 1.], [0., 0., 1.]])

cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
cross_entropy2 = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=y_))

with tf.Session() as sess:
    softmax = sess.run(y)
    c_e = sess.run(cross_entropy)
    c_e2 = sess.run(cross_entropy2)

    print("softmax result=", softmax)
    print("cross_entropy =", c_e)
    print("softmax_cross_entropy_with_logits=", c_e2)
