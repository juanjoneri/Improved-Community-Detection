#sys.path.append('C:/Users/juanj/Projects/LMU-RSCH-Fall17/Code/Python/my_packages')
import os
import sys
import numpy as np
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from my_packages.clusters.save_cluster import import_example
from my_packages.clusters.plot_cluster import plot_G
from my_packages.clusters.nx_np import nx_np

'''
# Description
This file showcases some useful operations that will be used for the algorithm
Can be thought of as a galery of tensorflow nodes

# On how to structure the project
https://www.tensorflow.org/versions/r0.12/get_started/basic_usage
'''

# CONSTRUCTION PHASE
## assemble the graph

# Constant Nodes
one = tf.constant(1.)
R = tf.constant(2)
n = tf.constant(20)

### source nodes (do not need any input i.e. Constant)
#R = tf.placeholder(tf.int32, name='n_clusters')
W = tf.placeholder(tf.float32, name='W') # The input adj matrix with 0s and 1s
F_i = tf.placeholder(tf.int32, name='initial_parition')

clas_i = tf.placeholder(tf.int32, name='target_clas') # value in [0, R-1]
node_i = tf.placeholder(tf.int32, name='node_index') # value in [0, n-1]

### Operation nodes
#n = tf.shape(W)[0]
D = tf.diag(tf.reduce_sum(W, 0), name='degree')
D_ = tf.diag((tf.pow(tf.diag_part(D), -0.5)))
I = tf.eye(n, name='identity')
L = tf.subtract(I, tf.matmul(D_, tf.matmul(W, D_)), name='Laplacian')

### Model nodes (to be trained)
F = tf.Variable(tf.fill([n, R], 0), name='Partition')
clas = tf.scatter_update(tf.Variable(tf.zeros([R])), [clas_i], one)
#clas = tf.assign(tf.zeros([R])[clas_i] , one) # gives [0, 0, 1] with a 1 on clas_i
#clas = tf.concat([tf.zeros(clas_i), [one], tf.zeros(R-clas_i-1)], 0) # to which class clas_i
#change_clas = tf.scatter_update(F, [node_i], clas) #what node?, to which class?
# sess.run(F.initializer)
# update F = F.assign(F + 1.0)



# EXCECUTION PHASE

if __name__ == '__main__':
    G, coordinates, labels_true = import_example('small')
    W_i, R_i = nx_np(G)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(sess.run(D, feed_dict={W:W_i}))
        print(sess.run(Ds, feed_dict={W:W_i}))
        print(sess.run(L, feed_dict={W:W_i}))
