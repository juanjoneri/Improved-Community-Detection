from my_packages.clusters.save_cluster import *
from my_packages.clusters.plot_cluster import plot_G

if __name__ == '__main__':
    H = import_cluster('./my_packages/clusters/examples/120-4/120n-4c-cluster.csv')
    h_coordinates, h_labels = import_metadata('./my_packages/clusters/examples/120-4/120n-4c-meta.csv')
    plot_G(H, h_coordinates)
    plot_G(H, h_coordinates, h_labels)
