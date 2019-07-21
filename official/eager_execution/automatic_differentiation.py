import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

tf.enable_eager_execution()

x=tf.ones((2,2))
def grad(x,persistent=False):   
    '''
    梯度计算。\n
    x:微分数据 \n
    persistent:是否开启持久\n
    梯度带API对于其输入变量自动微分\n
    还可以计算中间值的梯度(需要开启持久梯度带)\n
    默认情况下，只要调用GradientTape.gradient()方法，GradientTape所持有的资源就会被释放。
    要在同一计算中计算多个梯度，请创建一个持久梯度带。
    这允许对gradient()方法进行多次调用。
    ''' 
    with tf.GradientTape(persistent=persistent) as t:
        t.watch(x)
        z,y=math_op(x)
    dz_dx=t.gradient(z,x)
    print(dz_dx)
    
    # 中间值y的梯度  
    if persistent:
        dz_dy=t.gradient(z,y)   
        print(dz_dy)

def math_op(x):
    y=tf.reduce_sum(x)
    z=tf.multiply(y,y)
    return z,y



def control_flow(x,y):
    output=1.0
    for i in range(y):
        if i>1 and i<5:
            output=tf.multiply(output,x)
            print('output:',output)
    return output

def control_grad(x,y):
    with tf.GradientTape() as t:
        t.watch(x)
        out=control_flow(x,y)
        print(out,x)
    return t.gradient(out,x)    

x=tf.convert_to_tensor(2.0)
# 表达式等式为：y=f(x)=x^3 --> y'=3x^2 
assert control_grad(x,6).numpy()==12.0
# 表达式等式为：y=f(x)=x^3 --> y'=3x^2 
assert control_grad(x,5).numpy()==12.0
# 表达式等式为：y=f(x)=x^2 --> y'=2x 
assert control_grad(x,4).numpy()==4.0



grad(x,True)

# dz_dy=t.gradient(y,x)
# print(dz_dx)
# print(dz_dy)
# print(x)
# print(y)
# print(z)