import tensorflow as tf
import parameter

tf.reset_default_graph()

v1=tf.get_variable('v1',[parameter.size1],initializer=tf.zeros_initializer)
v2=tf.get_variable('v2',[parameter.size2],initializer=tf.zeros_initializer)


def restore_by_vname(nameValue):
    '''
    指定要保存或加载的名称和变量
    '''
    saver=tf.train.Saver(nameValue)
    print(nameValue)
    init_op = None if nameValue==None else tf.global_variables_initializer()

    run_session(saver,init_op)

    
def run_session(saver,init_op):
    with tf.Session() as sess:
        if init_op!=None: sess.run(init_op)
        saver.restore(sess,parameter.save_path)
        print('model restored')

        print('v1:%s'%v1.eval())
        print('v2:%s'%v2.eval())

# restore_by_vname(None)
restore_by_vname({'v1':v1})