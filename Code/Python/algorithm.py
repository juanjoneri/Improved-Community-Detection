from my_packages.model.decorators import lazy_property
from my_packages.clusters.save_cluster import import_example
from my_packages.clusters.plot_cluster import plot_G
from my_packages.clusters.nx_np import nx_np

import numpy as np
import tensorflow as tf

class Algorithm:

    one = tf.constant(1.)
    cero = tf.constant(0.)

    def __init__(self, graph_W, partition_F, alpha):
        self.graph_W = graph_W
        self.partition_F = partition_F
        self.u = tf.Variable([[1.],[0.],[0.],[0.],[1.],[1.],[1.],[0.],[1.],[0.],[1.],[1.],[1.],[0.],[0.],[1.],[0.],[1.],[0.],[0.]])
        self.alpha = alpha
        self.difuse

    @lazy_property
    def difuse(self):
        W = self.graph_W
        D = tf.diag(tf.reduce_sum(W, 0), name='degree')
        D_ = tf.diag((tf.pow(tf.diag_part(D), -0.5)))
        I = tf.eye(20, name='identity')
        Op = tf.matmul(D_, tf.matmul(W, D_), name='smooth_operator')
        self.u = tf.assign(self.u, tf.scalar_mul(self.alpha, tf.matmul(Op, self.u)) + tf.scalar_mul((self.one - self.alpha), self.u))
        return self.u


if __name__ == '__main__':
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    G, coordinates, labels_true = import_example('small')
    small_W, small_R = nx_np(G)

    n_nodes = 20
    n_clusters = 2

    test_F = np.zeros((n_nodes, n_clusters))
    test_F[0, 0] = 1

    graph_W = tf.placeholder(tf.float32, [n_nodes, n_nodes])
    partition_F = tf.placeholder(tf.float32, [n_nodes, n_clusters])
    alpha = tf.constant(0.9)

    algorithm = Algorithm(graph_W, partition_F, alpha)

    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

    F = sess.run(algorithm.difuse, {graph_W: small_W, partition_F: test_F})
    print(F)
