from my_packages.model.decorators import lazy_property
from my_packages.clusters.save_cluster import import_example
from my_packages.clusters.plot_cluster import plot_G
from my_packages.clusters.nx_np import nx_np

import numpy as np
import tensorflow as tf

class Algorithm:

    one = tf.constant(1, dtype=tf.float64)
    cero = tf.constant(0, dtype=tf.float64)

    def __init__(self, W, F, R, a):
        # Properties
        self.W = W       # Graph's adj matrix
        self.n = int(self.W.get_shape()[1])                # Number of vertexes
        self.R = R                                         # Target number of communities
        self.F = F                                         # Current partition: initialized random
        self.H = tf.Variable(F.initialized_value())                                         # Heat bump: initialized to F

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
        return tf.assign(self.H, tf.scalar_mul(a, tf.matmul(Op, self.H)) + tf.scalar_mul((one - a), F))

    @property
    def threshold(self):
        indices = tf.argmax(self.H, axis=1)
        return tf.assign(self.F, tf.squeeze(tf.one_hot(tf.cast(indices, tf.int32), self.R, dtype=tf.float64)))

    @lazy_property
    def labels(self):
        return tf.argmax(self.F, axis=1)

    @lazy_property
    def cut(self):
        F_C = tf.ones_like(self.F) - self.F
        return tf.matmul(tf.transpose(F_C), tf.matmul(self.W, self.F))

    @lazy_property
    def apply_constraints(self):
        return tf.count_nonzero(self.F, 0)

def random_partition(n, R):
    indices = tf.random_uniform([1,n], minval=0, maxval=R)
    return tf.Variable(tf.squeeze(tf.one_hot(tf.cast(indices, tf.int32), R, dtype=tf.float64)))

if __name__ == '__main__':
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    G, coordinates, labels_true = import_example('small')
    small_W, small_R = nx_np(G)

    n_nodes = 20
    n_clusters = small_R

    graph_W = tf.constant(small_W, dtype=tf.float64)
    graph_F = random_partition(20, 2)
    alpha = 0.9
    R = 2

    algorithm = Algorithm(graph_W, graph_F, R, alpha)

    # make sure the session is closed
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        ini_F = sess.run(algorithm.F)
        print('Initial F', ini_F)

        current_H = sess.run(algorithm.diffuse)
        print('current H', current_H)
        current_F = sess.run(algorithm.F)
        print('current F', current_F)

        th = sess.run(algorithm.threshold)
        end_F = sess.run(algorithm.F)
        end_H = sess.run(algorithm.H)
        print('th', th)
        print('F after Th', end_F)
        print('h after th', end_H)

        constraint = sess.run(algorithm.apply_constraints)
        print('constraints',constraint)

        print()
