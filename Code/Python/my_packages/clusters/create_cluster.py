import numpy as np
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
import networkx as nx
import csv

def init_nodes(n_nodes, centers, std):
    """
    Creates n nodes from a gaussian distribution arround the centers

    # Inputs
    n_nodes: the number of nodes in total
    centers: a list with 2 element lists with the coordinates of the centers like [[x1, y1], [x2, y2], [x3, y3] ...]
    std: the std distance of the nodes to their respective center

    # Outputs
    nodes / P: coordinates as np.array
    labels_true: the class they belong to (center)
    """
    nodes, labels_true = make_blobs(n_samples=n_nodes, centers=centers, cluster_std=std)
    return nodes, labels_true


def distance(p1, p2):
    """
    Calculate the distance between two nodes

    # Inputs
    p1 / p2: np arrays representing the coordinates
    """
    return np.linalg.norm(p2-p1)

def avg_center_distance(centers):
    '''
    # Inputs
    centers: a list with coordinates (lists of x, y lists)

    # Outupts
    the average distance between the centers in the list
    '''
    if len(centers) == 1: return 1
    center_pairs = zip(centers[0:], centers[1:])
    distances = []
    for c1, c2 in center_pairs:
        distances.append(distance(np.array(c1), np.array(c2)))
    return np.mean(distances)

def cluster(nodes, k):
    '''
    Create an matrix out of an np.array coordinates with coordinates
    will create a connection between any two nodes that are within k distance apart
    If no node is closer than k to a certain node, a connection will be created
    with the closest neighbor

    # Inputs
    nodes: coordinates as np.array
    k: a float value with the max distance of an edge

    # Outputs
    G: an nx graph with the newly created edges

    # Note
    coordinates are lost in this operation, nodes are stored as integers:
    0, 1, 2, 3 ... len(nodes)
    '''
    n = nodes.shape[0]
    G = nx.Graph()
    G.add_nodes_from(list(range(n)))

    for i in range(n):
        closest_node = -1
        closest_distance = float("inf")
        for j in range(n):
            if i == j: continue
            d = distance(nodes[i], nodes[j])
            if d <= k:
                G.add_edge(i, j)
            if d <= closest_distance:
                closest_distance = d
                closest_node = j
        G.add_edge(i, closest_node)
    return G

def create_cluster(n_nodes, centers, std, k):
    '''
    Combine the operations of `init_nodes` and `cluster` to fully create a connected graph.
    This function should be used in general for applications, and not the above two

    # Inputs
    n_nodes, centers, std: as specified by `init_nodes`
    k: as specified by `cluster`

    # Outputs
    G: a graph with connections drawn between nodes closer than k apart
    coordinates: a dict that maps to the cooordinates of each node, for plotting
    labels: the class each node belongs to
    '''
    nodes, labels = init_nodes(n_nodes, centers, std )
    G = cluster(nodes, k)
    coordinates = dict(list(enumerate(map(tuple, nodes)))) # a dict that maps to the cooordinates of each node, for plotting
    return G, coordinates, labels

if __name__ == '__main__':
    '''
    Sample call
    python3 create_cluster 90 ((0,0)(0,1)(1,0))
    '''
    import sys
    import re
    from save_cluster import export_cluster, export_metadata
    from plot_cluster import plot_G

    float_pat = r'([-+]?[0-9]+\.?[0-9]*)'

    n_nodes = int(sys.argv[1])
    c = list(map(float, re.findall(float_pat, sys.argv[2])))
    if n_nodes < 2 or len(c) % 2 != 0:
        print('try: python3 create_cluster 90 ((-1.0, 0.98)(-40, 13)(10.33, 0))')
        sys.exit()
    centers = [(c[i], c[i+1]) for i in range(0, len(c), 2)]
    d = avg_center_distance(centers)
    G, coordinates, labels = create_cluster(n_nodes, centers, std=d/4.5, k=d/2.5)

    plot_G(G, coordinates, labels)
    plot_G(G, coordinates)
    export_cluster(G, '{}n-{}c'.format(n_nodes, len(centers)))
    export_metadata(labels, coordinates, '{}n-{}c'.format(n_nodes, len(centers)))
