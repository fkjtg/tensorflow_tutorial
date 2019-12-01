"""
 axis参数详解：https://blog.csdn.net/wangying19911991/article/details/73928172
 较全参数测试用例：http://www.cnblogs.com/gaofighting/p/9671562.html
 重点在于对维度方面的理解：

"""

import tensorflow as tf

"""
 该数据的形状为2行3列shape=(2,3)，它属于二维数据，第一维的长度是2，第二维的长度是3。
 参数axis的值对应的是shape元组的索引
 输出的数据为原始数据的降维，形状则为原始形状删掉该索引数据
 快速识别数据的维数的2种方法：
   其一，计算shape元组的长度；
   其二，数一下数据最左侧连续左中括号的个数；  
"""
x = tf.constant([[1., 2., 3.], [4., 5., 6.]])
# x = tf.constant([[1., 2.], [1., 2.]])

# 当指定axis时，输出的数据形状为原数据剩下的形状
mean_none = tf.reduce_mean(x)
mean_kd_t = tf.reduce_mean(x, keep_dims=True)
mean_0 = tf.reduce_mean(x, 0)
mean_1 = tf.reduce_mean(x, 1)

# sess = tf.Session()
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
value_none = sess.run(mean_none)
value_kd_t = sess.run(mean_kd_t)
value_0 = sess.run(mean_0)
value_1 = sess.run(mean_1)

print("shape_x:", x.get_shape())  # (2, 3)

print("value_n", value_none)  # 3.5
print("value_kd_t", value_kd_t)  # [[3.5]]

print("value_0:", value_0)  # [2.5 3.5 4.5]
print("shape_0:", value_0.shape)  # (3,)

print("value_1:", value_1)  # [2. 5.]
print("shape_1:", value_1.shape)  # (2,)
