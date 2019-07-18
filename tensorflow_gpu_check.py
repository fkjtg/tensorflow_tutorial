import tensorflow as tf

print(tf.__version__)

cfig =tf.ConfigProto(log_device_placement=True)
sess = tf.Session(config=cfig)
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))