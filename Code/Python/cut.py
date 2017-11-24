import sys
sys.path.append('C:/Users/juanj/Projects/LMU-RSCH-Fall17/Code/Python/my_packages')
import numpy as np
from my_packages.clusters.save_cluster import import_example
from my_packages.clusters.plot_cluster import plot_G
from my_packages.clusters.nx_np import nx_np

if __name__ == '__main__':
    G, coordinates, labels = import_example('small')
    # This can be used to populate a tensorflow placeholder
    W, n = nx_np(G)
    print(W)
