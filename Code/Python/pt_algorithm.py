import torch
import numpy as np

from my_packages.clusters.save_cluster import import_example, import_cluster, import_metadata
from my_packages.clusters.plot_cluster import plot_G
from my_packages.clusters.nx_np import nx_np

class Algorithm:

    def __init__(self, W, R, a, constraints):
        self.W = W                                  # Graph's adj matrix
        self.R = R                                  # Target number of communities
        self.a = torch.FloatTensor([a])             # Alpha: diffusion parameter
        self.constraints = constraints              # (min, max) tuple with constraints for sizes of partition

        self.n = self.W.size()[0]                   # Number of vertexes
        self.F = self.random_partition(self.n, R)   # Initiate with a random partition
        self.D = torch.sum(self.W, 0)               # Degree vector(connections per node)
        self.H = self.F.clone()                     # Heat bump: initialized to F

        if (self.n - (R-1)*constraints[1] < constraints[0]):
            print("Alert: Not well defined constraints.")

    @property
    def C(self):
        # vector with name of the classes, or R if not assigned
        C = torch.zeros(self.n, 1)
        for row_index in range(self.n):
            row = self.F[row_index]
            if row.byte().any():
                C[row_index] = torch.max(row, dim=0)[1]
            else:
                C[row_index] = 4 # default to white for unnalocated classes
        return C

    @staticmethod
    def random_partition(n, R):
        # Creates a random partition with equal number of nodes in each class n/R
        C = torch.fmod(torch.randperm(n), R)
        i = torch.cat((torch.arange(n).type(torch.LongTensor), C), 0).view(2, n) #2D tensor with coordinates of values
        v = torch.ones(n) # 1D tesnor with values
        return torch.sparse.FloatTensor(i, v).to_dense()

    def diffuse(self, iterations):
        # Apply one diffuse step to the current heat distribution H
        I = torch.eye(self.n)
        D_ = torch.diag(torch.pow(self.D, -0.5))
        Op = torch.mm(D_, torch.mm(self.W, D_)).type(torch.FloatTensor)
        for _ in range(iterations):
            self.H = torch.mul(self.a, torch.mm(Op, self.H)) + torch.mul((1 - self.a), self.F)

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
        return (self.n - torch.nonzero(self.C - labels_true).size()[0])/self.n





if __name__ == '__main__':

    G = import_cluster("my_packages/clusters/examples/hm-1200/hm-1200n-cluster.csv")
    coordinates, labels_true = import_metadata("my_packages/clusters/examples/hm-1200/hm-1200n-meta.csv")
    small_W, small_R = nx_np(G)

    n_nodes = 1200
    n_clusters = 2

    graph_W = torch.from_numpy(small_W)

    algorithm = Algorithm(W=graph_W, R=n_clusters, a=0.99, constraints=(590, 610))

    iteration = 1
    plot_G(G, coordinates, algorithm.C, file_name="0A-random-partition")
    algorithm.reseed(1)
    plot_G(G, coordinates, algorithm.C, file_name="0B-1-seed".format(iteration))

    for seed_count in range(2, 200, 10):
        algorithm.diffuse(30)
        iteration += 1
        algorithm.rank_threshold()
        plot_G(G, coordinates, algorithm.C, file_name="{}A-rank_threshold".format(iteration, seed_count))
        algorithm.reseed(seed_count)
        plot_G(G, coordinates, algorithm.C, file_name="{}B-{}-seed".format(iteration, seed_count))

    algorithm.diffuse(30)
    algorithm.rank_threshold()
    plot_G(G, coordinates, algorithm.C, file_name="final-partition")

    #
    # print(algorithm.accuracy(torch.from_numpy(labels_true)))
    # plot_G(G, coordinates, algorithm.C)
    # # plot_G(G, coordinates, labels_true)
