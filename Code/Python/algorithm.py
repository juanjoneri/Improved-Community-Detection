from my_packages.model.decorators import lazy_property
from my_packages.clusters.save_cluster import import_example
from my_packages.clusters.plot_cluster import plot_G
from my_packages.clusters.nx_np import nx_np

import numpy as np
import tensorflow as tf
import random

class Algorithm:

    one = tf.constant(1., dtype=tf.float64)
    cero = tf.constant(0., dtype=tf.float64)

    def __init__(self, W, R, a):
        # Properties
        self.W = tf.constant(W, dtype=tf.float64)     # Graph's adj matrix
        self.n = int(self.W.get_shape()[1])           # Number of vertexes
        self.F = self.create_partition(self.n, R)     # Current partition: initialized random
        self.H = tf.Variable(self.F)                  # Heat bump: initialized to F
        self.a = tf.constant(a, dtype=tf.float64)     # Alpha: diffusion parameter
        # try to  change to int
        self.R = tf.constant(R, dtype=tf.float64)     # Target number of communities
        # Paremeters
        self.I = tf.eye(self.n, name='Identity')
        self.D = tf.diag(tf.reduce_sum(self.W, 0), name='Degree')
        D_ = tf.diag((tf.pow(tf.diag_part(self.D), -0.5)))
        self._Op = tf.matmul(D_, tf.matmul(self.W, D_), name='smooth_operator')

    @lazy_property
    def diffuse(self):
        W, F = self.W, self.F
        Op = self._Op
        cero, one, a = self.cero, self.one, self.a
        self.H = tf.assign(self.H, tf.scalar_mul(a, tf.matmul(Op, self.H)) + tf.scalar_mul((one - a), F))
        return self.H

    def create_group(self, R, r):
        group = np.zeros((1,R))
        group[0,r] = 1
        return tf.Variable(group, dtype=tf.float64)

    def greate_random_group(self, R):
        return self.create_group(R, int(random.random()*R))

    def create_partition(self, n, R):
        partition = self.greate_random_group(R)
        for _ in range(1, n):
            partition = tf.concat([partition, self.greate_random_group(R)], 0)
        return partition

if __name__ == '__main__':
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    G, coordinates, labels_true = import_example('small')
    small_W, small_R = nx_np(G)

    n_nodes = 20
    n_clusters = 2

    #graph_W = tf.placeholder(tf.float64, [n_nodes, n_nodes])
    graph_W = tf.constant(small_W, dtype=tf.float64)
    alpha = 0.9
    R = 3

    algorithm = Algorithm(small_W, R, alpha)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())


    rank = sess.run(algorithm.diffuse)
    print(rank)
    rank = sess.run(algorithm.diffuse)
    print(rank)
    final_F = sess.run(algorithm.F)
    print(final_F)
    final_H = sess.run(algorithm.H)
    print(final_H)
