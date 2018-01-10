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

    def diffuse(self):
        # Apply one diffuse step to the current heat distribution H
        I = torch.eye(self.n)
        D_ = torch.diag(torch.pow(self.D, -0.5))
        Op = torch.mm(D_, torch.mm(self.W, D_)).type(torch.FloatTensor)
        self.H = torch.mul(self.a, torch.mm(Op, self.H)) + torch.mul((1 - self.a), self.F)

    def reseed(self, seeds):
        #Take the k (`seeds`) best seeds from each heat bump specified by H
        pass

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
    initial_F[12][1] = 1

    algorithm = Algorithm(graph_W, initial_F, R=n_clusters, a=0.9)
    algorithm.diffuse()

    plot_G(G, coordinates)
