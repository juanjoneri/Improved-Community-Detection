import torch

from my_packages.clusters.save_cluster import import_example
from my_packages.clusters.plot_cluster import plot_G
from my_packages.clusters.nx_np import nx_np

class Algorithm:

    def __init__(self, W, F, R, a):
        self.W = W                      # Graph's adj matrix
        self.R = R                      # Target number of communities
        self.F = F                      # Indicator of current partition

        self.D = torch.sum(self.W, 0)   # Degree vector(connections per node)
        self.H = F.clone()              # Heat bump: initialized to F

        self.n = self.W.size()[0]       # Number of vertexes
        self.a = torch.FloatTensor([a])   # Alpha: diffusion parameter

    def diffuse(self, iterations):
        # Apply one diffuse step to the current heat distribution H
        I = torch.eye(self.n)
        D_ = torch.diag(torch.pow(self.D, -0.5))
        Op = torch.mm(D_, torch.mm(self.W, D_)).type(torch.FloatTensor)
        for _ in range(iterations):
            self.H = torch.mul(self.a, torch.mm(Op, self.H)) + torch.mul((1 - self.a), self.F)

    def reseed(self, seeds):
        # Take the k (`seeds`) best seeds from each heat bump specified by H
        allocated = set() # Track allocated nodes
        nodes_per_class = dict(zip(range(self.R), [0]*self.R)) # Track full classes

        ranks = torch.topk(self.H, self.n, dim=0)[1]
        new_F = torch.zeros(n_nodes, n_clusters)

        i = 0
        for rank in ranks:
            j = 0
            for node in rank:
                if node not in allocated and nodes_per_class[j] < seeds:
                    allocated.add(node)
                    nodes_per_class[j] += 1
                    new_F[node][j] = 1
                j += 1
            i += 1

        print(ranks)
        print(allocated)
        print(new_F)


    def labels(self):
        # Return a vector with labels for each class (only makes sense when F represents a partition)
        pass


if __name__ == '__main__':
    G, coordinates, labels_true = import_example('small')
    small_W, small_R = nx_np(G)

    n_nodes = 20
    n_clusters = 2

    graph_W = torch.from_numpy(small_W)
    initial_F = torch.zeros(n_nodes, n_clusters)
    initial_F[1][0] = 1
    initial_F[9][1] = 1

    algorithm = Algorithm(graph_W, initial_F, R=n_clusters, a=0.9)
    algorithm.diffuse(10)

    print(algorithm.reseed(6))
    plot_G(G, coordinates)
