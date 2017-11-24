import networkx as nx
import csv
import numpy as np
import pandas as pd
'''
# Description
Use this module for saving and retreiving graphs, its connections, coordinates and labels.
'''

def export_cluster(G, file_name):
    '''
    Writes the contents of a graph as an adjacency list to a file

    # Inputs:
    G: a graph to save
    file_name: the name of the file .csv to write to
    '''
    with open('{}-cluster.csv'.format(file_name), 'w') as result:
        writer = csv.writer(result, dialect='excel')
        writer.writerows(G.edges)

def export_metadata(coordinates, labels, file_name):
    '''
    Writes the ground truth of the classes of the graph to filname.txt as well as the coordinates of each point

    # Inputs:
    labels: the ground_truth of a graph to save
    coordinates: map of pairs of coordinates
    file_name: the name of the file .csv to write to
    '''
    with open('{}-meta.csv'.format(file_name), 'w') as result:
        for i in range(len(labels)):
            result.write('{}, {}, {}, {}\n'.format(i, labels[i], *coordinates[i]))

def import_cluster(file_name):
    data = pd.read_csv(file_name, header=None).values
    x, y = data[:,0], data[:,1]
    G = nx.Graph()
    G.add_nodes_from(range(np.max(x)))
    G.add_edges_from(zip(x, y))
    return G

def import_metadata(file_name):
    data = pd.read_csv(file_name, header=None).values
    xs, ys = data[:,2], data[:,3]
    coordinates = dict(zip(range(len(data)), zip(xs, ys)))
    labels = data[:,1]
    return coordinates, labels


if __name__ == '__main__':
    from create_cluster import *
    from plot_cluster import plot_G

    centers = [[0,0], [1,0], [0,1], [1,1]]
    d = avg_center_distance(centers)
    G, g_coordinates, g_labels = create_cluster(n_nodes=120, centers=centers, std=d/5, k=d/5)

    plot_G(G, g_coordinates, g_labels)
    print(g_coordinates)
    export_cluster(G, 'demo_W')
    export_metadata(g_coordinates, g_labels, 'demo_W')
    H = import_cluster('demo_W-cluster.csv')
    h_coordinates, h_labels = import_metadata('demo_W-meta.csv')
    print(h_coordinates)
    plot_G(H, h_coordinates, h_labels)