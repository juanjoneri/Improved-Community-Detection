#!/usr/bin/python3

from create_cluster import create_cluster, avg_center_distance, export_cluster
from plot_cluster import plot_G

"""
Demo for cluster Creation
-------------------------

Creates a cluster with 4 centers using the create_cluster library
Plots the output using the plot_cluster library
"""

if __name__ == '__main__':
    centers = [[0,0], [1,0], [0,1], [1,1]]
    d = avg_center_distance(centers)
    G, coordinates, labels = create_cluster(n_nodes=120, centers=centers, std=d/5, k=d/5)

    export_cluster(G, 'demo_W')
    plot_G(G, coordinates, labels)