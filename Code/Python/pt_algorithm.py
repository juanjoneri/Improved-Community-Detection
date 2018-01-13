import torch
import numpy as np

from my_packages.clusters.save_cluster import import_example
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
        # Take the k (`seeds`) best seeds from each heat bump specified by H
        allocated = set() # Track allocated nodes
        nodes_per_class = dict(zip(range(self.R), [0]*self.R)) # Track full classes
        max_nodes_per_class = self.constraints[1]

        ranks = torch.topk(self.H, self.n, dim=0)[1] # Order classes by heat score
        # self.F = torch.zeros(n_nodes, n_clusters) # Reset the partition

        i = 0
        for rank in ranks: # 1st places, 2nd places ... Rth places
            # Give class' rth choice to a random class, unless nodes is alocated or class full
            class_order = np.arange(self.R)
            np.random.shuffle(class_order) # pytorch has no shuffle
            for class_index in class_order:
                node = rank[class_index]
                if node not in allocated and nodes_per_class[class_index] <= max_nodes_per_class:
                    allocated.add(node)
                    nodes_per_class[class_index] += 1
                    self.F[node][class_index] = 1
            i += 1

    @property
    def labels(self):
        # Return a vector with labels for each class (only makes sense when F represents a partition)
        return torch.max(self.F, dim=1)[1]


if __name__ == '__main__':
    G, coordinates, labels_true = import_example('big')
    small_W, small_R = nx_np(G)

    n_nodes = 120
    n_clusters = 4

    graph_W = torch.from_numpy(small_W)
    initial_F = torch.zeros(n_nodes, n_clusters)
    initial_F[1][0] = 1
    initial_F[2][1] = 1
    initial_F[3][2] = 1
    initial_F[4][3] = 1

    algorithm = Algorithm(W=graph_W, F=initial_F, R=n_clusters, a=0.9, constraints=(25, 35))
    algorithm.diffuse(10)

    algorithm.rank_threshold()
    plot_G(G, coordinates, algorithm.labels)
