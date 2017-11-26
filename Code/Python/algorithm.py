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
    iterations = 50 # u^infinity

    def __init__(self, graph_W, alpha):
        self.graph_W = graph_W
        self.partition_F = self.create_partition(20, 3)
        self.U = tf.Variable(self.partition_F)
        self.alpha = alpha
        self.difuse

    @lazy_property
    def difuse(self):
        W = self.graph_W
        D = tf.diag(tf.reduce_sum(W, 0), name='degree')
        D_ = tf.diag((tf.pow(tf.diag_part(D), -0.5)))
        I = tf.eye(20, name='identity')
        Op = tf.matmul(D_, tf.matmul(W, D_), name='smooth_operator')
        for _ in range(self.iterations):
            self.U = tf.assign(self.U, tf.scalar_mul(self.alpha, tf.matmul(Op, self.U)) + tf.scalar_mul((self.one - self.alpha), self.partition_F))
        return self.U

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

    graph_W = tf.placeholder(tf.float64, [n_nodes, n_nodes])
    alpha = tf.constant(0.9, dtype=tf.float64)

    algorithm = Algorithm(graph_W, alpha)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    initial_F = sess.run(algorithm.partition_F, {graph_W: small_W})
    print(initial_F)
    rank = sess.run(algorithm.difuse, {graph_W: small_W})
    print(rank)
    final_F = sess.run(algorithm.partition_F, {graph_W: small_W})
    print(final_F)
