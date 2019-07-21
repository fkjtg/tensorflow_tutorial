import os
import tensorflow as tf
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

tf.enable_eager_execution()

def init_var():    
    x=tf.zeros([10,10])
    x+=2
    print(x)

def assign_value():
    v=tf.contrib.eager.Variable(1.0)
    print(v)

    v.assign(3.0)
    print(v)

    v.assign(tf.square(v))
    print(v.numpy())

# assign_value()



class Model(object):
    def __init__(self):
        self.W=tf.contrib.eager.Variable(5.0)
        self.b=tf.contrib.eager.Variable(0.0)

    def __call__(self,x):
        return self.W*x+self.b

def model_test():
    '''
    对模型进行实例测试
    '''
    model= Model()
    result = model(3.0)
    print(result)

def loss(predicted_y,desired_y):
    '''
    损失函数，用来对比模型计算出来的值与欲望值
    '''
    return tf.reduce_mean(tf.square(predicted_y-desired_y))

def train(model,inputs,outputs,learning_rate):
    with tf.GradientTape() as t:
        current_loss=loss(model(inputs),outputs)    
    dw,db=t.gradient(current_loss,[model.W,model.b])
    model.W.assign_sub(learning_rate*dw)
    model.b.assign_sub(learning_rate*db)

def train_test():
    model =Model()
    Ws,bs=[],[]
    epochs=range(10)
    for epoch in epochs:
        Ws.append(model.W.numpy())
        bs.append(model.b.numpy())
        pre_y=model(inputs)
        current_loss=loss(pre_y,outputs)
        
        train(model,inputs,outputs,learning_rate=0.1)
        print('Epoch %2d: W=%1.2f b=%1.2f,loss=%2.5f'%(epoch,Ws[-1],bs[-1],current_loss))
    
    len_epochs=len(epochs)
    plt.plot(   epochs,Ws,'r',
                epochs,bs,'b')
    plt.plot(   [TRUE_W]*len_epochs,'r--',
                [TRUE_b]*len_epochs,'b--')
    plt.legend(['W','b','true W','true b'])
    plt.show()


TRUE_W=3.0
TRUE_b=2.0
NUM_EXAMPLES=1000
inputs = tf.random_normal(shape=[NUM_EXAMPLES])
noises = tf.random_normal(shape=[NUM_EXAMPLES])
outputs=inputs*TRUE_W+TRUE_b+noises


def original_test():  
    '''
    用模型对初始数据显示
    '''
    model = Model()
    y=model(inputs)
    plt.scatter(inputs,outputs,c='b')
    plt.scatter(inputs,y,c='r')
    print('current loss:')
    print(loss(y,outputs).numpy())
    plt.show()

# train_test()
original_test()

