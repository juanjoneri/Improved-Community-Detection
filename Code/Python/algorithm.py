from my_packages.model.decorators import define_scope
from my_packages.clusters.save_cluster import import_example
from my_packages.clusters.plot_cluster import plot_G
from my_packages.clusters.nx_np import nx_np

import numpy as np
import tensorflow as tf

class Algorithm:

    one = tf.constant(1, dtype=tf.float64)
    cero = tf.constant(0, dtype=tf.float64)

    def __init__(self, W, F, R, a):
        self.W = W                                         # Graph's adj matrix
        self.n = int(self.W.get_shape()[1])                # Number of vertexes
        self.R = R                                         # Target number of communities
        self.F = F                                         # Current partition: initialized random
        self.H = tf.Variable(F.initialized_value())        # Heat bump: initialized to F
        self.a = tf.constant(a, dtype=tf.float64)          # Alpha: diffusion parameter

    @define_scope
    def diffuse(self):
        cero, one, a = self.cero, self.one, self.a
        W, n = self.W, self.n
        # Preliminary operations, only cimputed once thanks to decorator (note it is indep of step)
        I = tf.eye(n, name='Identity')
        D = tf.diag(tf.reduce_sum(W, 0), name='Degree')
        D_ = tf.diag((tf.pow(tf.diag_part(D), -0.5)), name='D_pow')
        Op = tf.matmul(D_, tf.matmul(W, D_), name='W_D_pow_W')
        return tf.assign(self.H, tf.scalar_mul(a, tf.matmul(Op, self.H)) + tf.scalar_mul((one - a), self.F))

    @define_scope
    def threshold(self):
        indices = tf.argmax(self.H, axis=1)
        return tf.assign(self.F, tf.squeeze(tf.one_hot(tf.cast(indices, tf.int32), self.R, dtype=tf.float64)))

    @define_scope
    def labels(self):
        return tf.argmax(self.F, axis=1)

    @define_scope
    def cut(self):
        return tf.matmul(tf.transpose(self.F), tf.matmul(self.W, self.F))

    @define_scope
    def apply_constraints(self):
        return tf.count_nonzero(self.F, 0)

def random_partition(n, R):
    indices = tf.random_uniform([1,n], minval=0, maxval=R)
    return tf.Variable(tf.squeeze(tf.one_hot(tf.cast(indices, tf.int32), R, dtype=tf.float64)))

def test(G, coordinates, algorithm, plot=False):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        ini_F = sess.run(algorithm.F)
        ini_S = sess.run(algorithm.apply_constraints)
        print('Initial F\n', ini_F)
        print('Initial sizes\n', ini_S)

        for i in range(5):
            print('\n# Diffusion pass {}\n'.format(i+1))
            for _ in range(10):
                sess.run(algorithm.diffuse)

            H = sess.run(algorithm.H)
            print('new H\n', H)
            sess.run(algorithm.threshold)
            F = sess.run(algorithm.F)
            print('new F\n', F)
            if plot:
                current_labels = sess.run(algorithm.labels)
                plot_G(G, coordinates, current_labels)

        print('\n# Finally\n')
        count = sess.run(algorithm.apply_constraints)
        labels = sess.run(algorithm.labels)
        print('labels\n', labels)
        print('count\n ',count)

        print('\n# Actual Labels')
        print(labels_true.astype(int))

if __name__ == '__main__':
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    G, coordinates, labels_true = import_example('small')
    small_W, small_R = nx_np(G)

    n_nodes = 20
    n_clusters = 2

    graph_W = tf.constant(small_W, dtype=tf.float64)
    graph_F = random_partition(n_nodes, n_clusters)
    alpha = 0.9
    R = n_clusters

    algorithm = Algorithm(graph_W, graph_F, R, alpha)
    test(G, coordinates, algorithm, plot=False)
