import tensorflow as tf
import numpy as np
import tempfile
import time

def eager_exe_enable_or_not():   
    '''
    `enable_eager_execution`开启与未开的区别在于输出是否能看到结果
    '''
    # print(tf.add(1,2))
    # output-->:Tensor("Add:0", shape=(), dtype=int32)

    tf.enable_eager_execution()
    print(tf.add(1,2))
    # output-->:tf.Tensor(3, shape=(), dtype=int32)

eager_exe_enable_or_not()

def np_compatibility():
    '''
    numpy与tensor的兼容能力
    '''
    # 将ndarray转换成tensor数据
    ndarray=np.ones([3,3])
    tensor=tf.multiply(ndarray,42)
    print(tensor)

    # 将tensor转换ndarray
    print(np.add(tensor,1))

    print(tensor.numpy())

def gpu_acceleration():
    '''
    `x.device.endswith("GPU:0")`张量指定gpu进行加速操作
    '''
    x=tf.random_uniform([3,3])
    print(tf.test.is_gpu_available())
    print(x.device.endswith('GPU:0'))

def time_matmul(x):
    '''
    记时乘法操作
    '''
    start=time.time()
    for loop in range(10):
        tf.matmul(x,x)    
    result=time.time()-start
    print('10 loops:{:0.2f}ms'.format(1000*result))


def with_execution(dn):
    '''
    张量根据设备名称，进行张量的乘法操作
    '''
    print(dn[:-1])
    with tf.device(dn):
        x = tf.random_uniform([1000,1000])
        assert x.device.endswith(dn)
        time_matmul(x)

def device_cpu_gpu():
    '''
    对比cpu计算与gpu计算
    '''
    with_execution('CPU:0')
    if tf.test.is_gpu_available:    
        with_execution('GPU:0')


def create_dataset():
    ds_tensors = tf.data.Dataset.from_tensor_slices([1, 2, 3, 4, 5, 6])

    # Create a CSV file   
    _, filename = tempfile.mkstemp()

    with open(filename, 'w') as f:
        f.write(
        """
        Line 1
        Line 2
        Line 3
        """)

    ds_file = tf.data.TextLineDataset(filename)
    print(ds_tensors)
    ds_tensors = ds_tensors.map(tf.square)
    print(ds_tensors)
    ds_tensors=ds_tensors.shuffle(2)
    print(ds_tensors)
    ds_tensors=ds_tensors.batch(3)
    print(ds_tensors)

    ds_file = ds_file.batch(2)

    print('\n elements of ds_tensors:')
    for x in ds_tensors:
        print(x)

    print('\n elements in ds_file: ')
    for x in ds_file:
        print(x)


# np_compatibility()
# gpu_acceleration()

# device_cpu_gpu()
# create_dataset()
