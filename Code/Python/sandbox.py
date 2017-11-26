import numpy as np
import tensorflow as tf





def create_group(R, r):
    group = np.zeros((1,R))
    group[0,r] = 1
    print(group)
    return tf.Variable(group, dtype=tf.float64)


if __name__ == '__main__':
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        g = sess.run(create_group(4,2))
        print(g)
