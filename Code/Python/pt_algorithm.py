import torch
import numpy as np

from my_packages.clusters.save_cluster import import_example, import_cluster, import_metadata
from my_packages.clusters.plot_cluster import plot_G
from my_packages.clusters.nx_np import nx_np

class Algorithm:

    def __init__(self, W, F, R, a, constraints):
        self.W = W                      # Graph's adj matrix
        self.R = R                      # Target number of communities
        self.F = F                      # Indicator of current partition
        self.constraints = constraints  # (min, max) tuple with constraints for sizes of partition

        self.D = torch.sum(self.W, 0)   # Degree vector(connections per node)
        self.H = F.clone()              # Heat bump: initialized to F

        self.n = self.W.size()[0]       # Number of vertexes
        self.a = torch.FloatTensor([a]) # Alpha: diffusion parameter

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
        max_nodes_per_class = self.constraints[1]
        nodes_per_class = dict(zip(range(self.R), [0]*self.R)) # Mantain a size constraint

        ranks = torch.topk(self.H, self.n, dim=0)[1] # Order classes by heat score
        new_F = torch.zeros(self.n, self.R) # Reset the partition

        for rank in ranks: # 1st, 2nd ... Rth places
            class_order = np.arange(self.R)
            np.random.shuffle(class_order)
            for class_index in class_order: # class 2, 9, ... randomly
                node = rank[class_index]
                if node not in allocated and nodes_per_class[class_index] <= max_nodes_per_class:
                    new_F[node][class_index] = 1
                    allocated.add(node)
                    nodes_per_class[class_index] += 1
        self.F = new_F
        self.H = self.F.clone()

    def random_threshold(self):
        nodes_per_class = dict(zip(range(self.R), [0]*self.R)) # Mantain a size constraint
        max_nodes_per_class = self.constraints[1]

        node_order = np.arange(self.n)
        np.random.shuffle(node_order)
        new_F = torch.zeros(self.n, self.R) # Reset the partition

        for node_index in node_order:
            node_heat = self.H[node_index]
            node_ranks = torch.topk(node_heat, self.R, dim=0)[1] # Order classes by heat score
            for class_choice in node_ranks:
                if nodes_per_class[class_choice] <= max_nodes_per_class:
                    new_F[node_index][class_choice] = 1
                    nodes_per_class[class_choice] += 1
                    break
        self.F = new_F
        self.H = self.F.clone()


    def reseed(self, seed_count):
        '''
        Select seed_count many seeds for each partition at random.
        Should only be applied after thresholding
        '''
        if seed_count > self.n / self.R:
            raise Exception('Too many seeds')

        nodes_per_class = dict(zip(range(self.R), [0]*self.R))
        node_order = np.arange(self.n)
        np.random.shuffle(node_order)
        new_F = torch.zeros(self.n, self.R) # Reset the partition
        for node_index in node_order:
            node_class = int(torch.nonzero(self.F[node_index])) # only has one element, the index of the nonzero element
            if nodes_per_class[node_class] < seed_count:
                nodes_per_class[node_class] += 1
                new_F[node_index][node_class] = 1
        self.F = new_F

    def error(self, labels_true):
        return (self.n - torch.nonzero(self.labels - labels_true).size()[0])/self.n


    @property
    def labels(self):
        # Return a vector with labels for each class as specified by the current Indicator matrix F
        return torch.max(self.F, dim=1)[1].type(torch.DoubleTensor)


if __name__ == '__main__':

    G = import_cluster("my_packages/clusters/examples/hm-1200/hm-1200n-cluster.csv")
    coordinates, labels_true = import_metadata("my_packages/clusters/examples/hm-1200/hm-1200n-meta.csv")
    small_W, small_R = nx_np(G)

    n_nodes = 1200
    n_clusters = 2

    graph_W = torch.from_numpy(small_W)
    initial_F = torch.zeros(n_nodes, n_clusters)
    initial_F[0][1] = 1
    initial_F[177][0] = 1

    algorithm = Algorithm(W=graph_W, F=initial_F, R=n_clusters, a=0.99, constraints=(590, 610))

    for seeds in range(1, 600, 10):
        algorithm.diffuse(30)
        algorithm.random_threshold()
        algorithm.reseed(seeds)
    algorithm.diffuse(30)
    algorithm.rank_threshold()

    print(algorithm.error(torch.from_numpy(labels_true)))
    plot_G(G, coordinates, algorithm.labels)
    # plot_G(G, coordinates, labels_true)
