import torch
import numpy as np
import scipy

from my_packages.clusters.save_cluster import *
from my_packages.clusters.plot_cluster import plot_G
from my_packages.clusters.nx_np import nx_np

class Algorithm:

    def __init__(self, W, R, n, a, constraints):
        self.W = W                                  # Graph's adj matrix (sparse)
        self.R = R                                  # Target number of communities
        self.n = n                                  # Number of vertices
        self.a = torch.FloatTensor([a])             # Alpha: diffusion parameter
        self.constraints = constraints              # (min, max) tuple with constraints for sizes of partition

        self.F = self.random_partition(self.n, R)   # Initiate with a random partition
        self.H = self.F.clone()                     # Heat bump: initialized to F
        self.Op = self._create_operator()

        if (self.n - (R-1)*constraints[1] < constraints[0]):
            print("Alert: Not well defined constraints.")

    @property
    def C(self):
        # pytorch vector with name of the classes, or 10 if not assigned
        C = torch.zeros(self.n, 1).type(torch.LongTensor)
        for row_index in range(self.n):
            row = self.F[row_index]
            if row.byte().any():
                C[row_index] = torch.max(row, dim=0)[1]
            else:
                C[row_index] = 10 # default to white for unnalocated classes
        return C

    @property
    def D(self):
        # pytorch degree vector of an undirected graph
        # nodes must all be connected
        i = self.W._indices()[0].numpy()
        x, y =  np.unique(i, return_counts=True)
        return torch.from_numpy(x[y])

    @staticmethod
    def random_partition(n, R):
        # Creates a random partition with equal number of nodes in each class n/R
        C = torch.fmod(torch.randperm(n), R)
        i = torch.cat((torch.arange(n).type(torch.LongTensor), C), 0).view(2, n) #2D tensor with coordinates of values
        v = torch.ones(n) # 1D tesnor with values
        return torch.sparse.FloatTensor(i, v).to_dense()

    def diffuse(self, iterations):
        # Apply one diffuse step to the current heat distribution H
        for _ in range(iterations):
            self.H = torch.mul(self.a, torch.mm(self.Op, self.H)) + torch.mul((1 - self.a), self.F)

    def _create_operator(self):
        # using scipy since pytorch is in betta and has no sparse X sparse yet
        self.D.numpy()
        D_pow = np.power(self.D.numpy(), -0.5)
        D_ = scipy.sparse.diags(D_pow)
        i = self.W._indices().numpy()
        v = self.W._values()
        W = scipy.sparse.coo_matrix((v, (i[0,:], i[1,:])), shape=(self.n, self.n))
        Op = D_ * W * D_
        return torch.sparse.FloatTensor(\
            torch.from_numpy(i).type(torch.LongTensor),\
            torch.from_numpy(Op.data).type(torch.FloatTensor))

    def rank_threshold(self):
        '''
        Oder the nodes by "temperature" for each class in an NxR rank vecotr, try to allocate each class their top choice
        by looping through the rank vector and allocating the nodes to each class if not already allocated or the class is full
        (each row in the rank vector is iterated randomly to not be bias to any class)
        '''
        allocated = set() # Track allocated nodes (nodes can only belong to one class)
        nodes_per_class = dict(zip(range(self.R), [0]*self.R)) # Mantain a size constraint
        max_nodes_per_class = self.constraints[1]

        ranks = torch.topk(self.H, self.n, dim=0)[1] # Order classes by heat score
        C = self.C.clone()
        self.F = torch.zeros(self.n, self.R)

        for rank in ranks: # 1st, 2nd ... Rth places
            class_order = torch.randperm(self.R).type(torch.LongTensor)
            for class_index in class_order: # class 2, 9, ... randomly
                node = rank[class_index]
                if node not in allocated and nodes_per_class[class_index] < max_nodes_per_class:
                    self.F[node][class_index] = 1
                    allocated.add(node)
                    nodes_per_class[class_index] += 1

        self.H = self.F.clone()

    def random_threshold(self):
        nodes_per_class = dict(zip(range(self.R), [0]*self.R)) # Mantain a size constraint
        max_nodes_per_class = self.constraints[1]
        self.F = torch.zeros(self.n, self.R)

        node_order = torch.randperm(self.n)
        for node_index in node_order:
            node_heat = self.H[node_index]
            node_ranks = torch.topk(node_heat, self.R, dim=0)[1] # Order classes by heat score
            for class_choice in node_ranks:
                if nodes_per_class[class_choice] < max_nodes_per_class:
                    self.F[node_index][class_choice] = 1
                    nodes_per_class[class_choice] += 1
                    break

        self.H = self.F.clone()


    def reseed(self, seed_count):
        '''
        Select seed_count many seeds for each partition at random.
        Should only be applied after thresholding
        '''
        if seed_count > self.n / self.R:
            raise Exception('Too many seeds')

        nodes_per_class = dict(zip(range(self.R), [0]*self.R))
        node_order = torch.randperm(self.n)
        for node_index in node_order:
            node_class = int(self.C[node_index]) # only has one element, the index of the nonzero element
            if nodes_per_class[node_class] < seed_count:
                nodes_per_class[node_class] += 1
                self.F[node_index][node_class] = 1
            else:
                self.F[node_index][node_class] = 0

        self.H = self.F.clone()

    def purity(self, labels_true):
        # from http://www.caner.io/purity-in-python.html
        A = np.c_[(self.C.numpy(), labels_true)].astype(int)
        n_accurate = 0.
        for j in np.unique(A[:,0]):
            z = A[A[:,0] == j, 1]
            x = np.argmax(np.bincount(z))
            n_accurate += len(z[z == x])
        return round(n_accurate / self.n * 100, 2)


if __name__ == '__main__':

    n_nodes = 20000
    n_clusters = 20

    W = import_sparse("../Experiments/20news/20news.csv")
    labels_true = import_labels("../Experiments/20news/20news_labels.csv")

    algorithm = Algorithm(W=W, R=n_clusters, n=n_nodes, a=0.9, constraints=(990, 1010))

    iteration = 1
    algorithm.reseed(1)
    algorithm.diffuse(15)
    algorithm.rank_threshold()
    print(iteration, algorithm.purity(labels_true))

    for seed_count in range(2, 300, 10):
        iteration += 1
        algorithm.diffuse(15)
        algorithm.rank_threshold()
        print(iteration, algorithm.purity(labels_true))
        algorithm.reseed(seed_count)

    algorithm.diffuse(15)
    algorithm.rank_threshold()
    print("final: ", algorithm.purity(labels_true))
