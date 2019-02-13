"""
测试思路：
    首先，搞清楚transpose是在做什么，各参数的作用；
    其次，搞清楚源数据的形状；
    接着，得到测试结果；
    最终，验证原理，保证理解透彻
"""

import numpy as np
import tensorflow as tf

a = [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]
a = np.array(a)

print(a)
print('shape:', a.shape)

# 对低维数据与高维数据进行灵活转置
# a的转置是根据 perm 的设定值来进行的。
# 返回数组的 dimension（尺寸、维度） i与输入的 perm[i]的维度相一致。
# 如果未给定perm，默认设置为 (n-1...0)，这里的 n 值是输入变量的 rank 。
# 因此默认情况下，这个操作执行了一个正规（regular）的2维矩形的转置
b = tf.transpose(a, [0, 2, 1])
with tf.Session() as sess:
    b = sess.run(b)
    print(b)
    print('shape:', b.shape)
