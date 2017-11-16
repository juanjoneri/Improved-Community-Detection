#!/usr/bin/python3

from create_cluster import create_cluster, avg_center_distance, export_cluster
from networkx_plot import plot_G
import csv

centers = [[0,0], [1,0], [0,1], [1,1]]
d = avg_center_distance(centers)
G, coordinates, labels = create_cluster(n_nodes=120, centers=centers, std=d/5, k=d/5)

export_cluster(G, 'W1')
plot_G(G, coordinates, labels)
