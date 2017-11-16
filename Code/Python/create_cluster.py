import numpy as np
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
import networkx as nx
from networkx_plot import plot_G
import csv


def init_nodes(n, centers, std):
    """
    Creates n nodes from a gaussian distribution arround the centers
    P: coordinates as np.array
    labels_true: the class they belong to (center)
    """
    P, labels_true = make_blobs(n_samples=n, centers=centers, cluster_std=std, random_state=0)
    return P, labels_true


def distance(p1, p2):
    return np.linalg.norm(p2-p1)

def avg_center_distance(centers):
    # from a bunch of centers, it will tell you the mean distance
    center_pairs = zip(centers[0:], centers[1:])
    distances = []
    for c1, c2 in center_pairs:
        distances.append(distance(np.array(c1), np.array(c2)))
    return np.mean(distances)

def cluster(P, k):
    n = P.shape[0]
    G = nx.Graph()
    G.add_nodes_from(list(range(n)))

    for i in range(1, n):
        closest_node = -1
        closest_distance = float("inf")
        for j in range(i):
            d = distance(P[i], P[j])
            if d <= k:
                G.add_edge(i, j)
            if d <= closest_distance:
                closest_distance = d
                closest_node = j
        G.add_edge(i, closest_node) # make sure nodes have at least one connection

    return G

def create_cluster(n_nodes, centers, std, k):
    P, labels = init_nodes(n_nodes, centers, std )
    G = cluster(P, k)
    coordinates = dict(list(enumerate(map(tuple, P)))) # a dict that maps to the cooordinates of each node, for plotting
    return G, coordinates, labels

def export_cluster(G, name):
    with open('{}.csv'.format(name), 'w') as result:
        writer = csv.writer(result, dialect='excel')
        writer.writerows(G.edges)

if __name__ == '__main__':
    print('Number of nodes')
    n_nodes = int(input())
    print('Number of clusters')
    n_clusters = int(input())

    if n_nodes < 2 or n_clusters < 2:
        print('Cannot build graph from that')
        sys.exit()
    centers = []
    for i in range(1, n_clusters+1):
        print('Center of cluster {} as x,y'.format(i))
        x, y = map(int, input().split(','))
        centers.append([x, y])

    G, coordinates, labels = create_cluster(n_nodes, centers, std=avg_center_distance(centers) / 3, k=avg_center_distance(centers) / 4)
    plot_G(G, coordinates, labels)
