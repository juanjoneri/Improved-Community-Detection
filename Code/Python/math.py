import numpy as np
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
"""
Test basic math using tensorflow graphs:
https://www.tensorflow.org/api_guides/python/math_ops#Arithmetic_Operators
"""

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

single_values = {X: 6, Y: 11}
list_values = {X: [1, 2, 3] ,Y: [9, 5, 2]}
numpy_values = {X: np.array([1, 2, 3]), Y: np.array([9, 5, 2])}

addition = tf.add(X, Y)
subtraction = X - Y
multiplication = tf.multiply(X, Y)

W = tf.Variable(np.array([[1,2], [3,4], [5,6]]), dtype=tf.float32, name='W_matrix') # in general this is the input of our program
                                                                  # so its a placeholder
S = tf.Variable(np.array([[0,0,1,1], [0,0,1,0], [1,1,0,0], [1,0,0,0]]), dtype=tf.float32, name='square_matrix')
w_init = W.initializer
s_init = S.initializer

# Get the degree matrix D of a square matrix S
degree_D = tf.diag( tf.reduce_sum(S, 0) )

# Control structure
i = tf.Variable(0, name="counter")
one = tf.constant(1)
global_init = tf.global_variables_initializer()

if __name__ == '__main__':

    with tf.Session() as sess:
        print('\nAddition')
        print('add single values', sess.run(addition, feed_dict=single_values))
        print('add list values', sess.run(addition, feed_dict=list_values))
        print('add np arrays', sess.run(addition, feed_dict=numpy_values))

        print('\nMultiplication')
        print('multiply single values', sess.run(multiplication, feed_dict=single_values))
        print('multiply list values', sess.run(multiplication, feed_dict=list_values))
        print('multiply np arrays', sess.run(multiplication, feed_dict=numpy_values))

        print('\nCreate a diagonal matrix:\n', sess.run(tf.diag(X), feed_dict=numpy_values))

        print('\nOperations with Matrices')
        sess.run(w_init)
        sess.run(s_init)
        print('W:\n', sess.run(W))
        print('S:\n', sess.run(S))
        print('W.t:\n', sess.run(tf.transpose(W)))
        print('2*S:\n', sess.run(tf.scalar_mul(2, S))) # should be a constant
        print('S^(-1/2):\n', sess.run(tf.pow(S, -0.5)))
        print('get the degree:\n', sess.run(degree_D))

        print('\nLoops:')
        sess.run(global_init) # now need those variables
        for _ in range(3):
            sess.run(tf.assign(i, tf.add(i, one)))
            print(sess.run(i), end=' ')
        print()
