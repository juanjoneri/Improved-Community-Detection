from my_packages.model.decorators import lazy_property
from my_packages.clusters.save_cluster import import_example
from my_packages.clusters.plot_cluster import plot_G
from my_packages.clusters.nx_np import nx_np

import numpy as np
import tensorflow as tf

class Algorithm:

    one = tf.constant(1, dtype=tf.float64)
    cero = tf.constant(0, dtype=tf.float64)

    def __init__(self, W, R, a):
        # Properties
        self.W = tf.constant(W, dtype=tf.float64)          # Graph's adj matrix
        self.n = int(self.W.get_shape()[1])                    # Number of vertexes
        self.R = R                           # Target number of communities
        self.F = self.random_partition()                   # Current partition: initialized random
        self.H = tf.Variable(self.F)                       # Heat bump: initialized to F

        # Operators
        self.a = tf.constant(a, dtype=tf.float64)          # Alpha: diffusion parameter
        self.I = tf.eye(self.n)                            # Identity matrix
        self.D = tf.diag(tf.reduce_sum(self.W, 0))         # Degree Matrix of W
        D_ = tf.diag((tf.pow(tf.diag_part(self.D), -0.5))) # D^(-1/2)
        self._Op = tf.matmul(D_, tf.matmul(self.W, D_))    # D^(-1/2) W D^(-1/2)

    @lazy_property
    def diffuse(self):
        W, F = self.W, self.F
        Op = self._Op
        cero, one, a = self.cero, self.one, self.a
        self.H = tf.assign(self.H, tf.scalar_mul(a, tf.matmul(Op, self.H)) + tf.scalar_mul((one - a), F))
        return self.H

    @lazy_property
    def threshold(self):
        indices = tf.argmax(self.H, axis=1)
        self.F = tf.squeeze(tf.one_hot(tf.cast(indices, tf.int32), self.R, dtype=tf.float64))
        return self.F

    @lazy_property
    def labels(self):
        return tf.argmax(self.F, axis=1)

    @lazy_property
    def cut(self):
        F_C = tf.ones_like(self.F) - self.F
        return tf.matmul(tf.transpose(F_C), tf.matmul(self.W, self.F))

    def random_partition(self):
        indices = tf.random_uniform([1,self.n], minval=0, maxval=self.R)
        return tf.squeeze(tf.one_hot(tf.cast(indices, tf.int32), self.R, dtype=tf.float64))

if __name__ == '__main__':
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    G, coordinates, labels_true = import_example('small')
    small_W, small_R = nx_np(G)

    n_nodes = 20
    n_clusters = small_R

    graph_W = tf.constant(small_W, dtype=tf.float64)
    alpha = 0.9
    R = 2

    algorithm = Algorithm(small_W, R, alpha)

    # make sure the session is closed
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        ini_cut = sess.run(algorithm.cut)
        print('Initial cut', ini_cut)

        for i in range(10):
            for _ in range(20):
                sess.run(algorithm.diffuse)
            current_F = sess.run(algorithm.threshold)
            current_Labels = sess.run(algorithm.labels)
            current_cut = sess.run(algorithm.cut)
            print('step', i, current_Labels, '\n' , current_cut)
            plot_G(G, coordinates, current_Labels)
            print()

        print(labels_true)
