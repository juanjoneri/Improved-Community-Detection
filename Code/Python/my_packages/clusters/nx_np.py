import numpy as np
import networkx as nx

'''
# Description
This module simplifies the translation of nx graphs into numpy arrays for use with tensorflow

# Info
https://networkx.github.io/documentation/networkx-1.7/reference/generated/networkx.convert.to_numpy_matrix.html
'''

def nx_np(G):
    n = G.order()
    A = nx.to_numpy_matrix(G, nodelist = list(range(n)))

    return A, n

def np_nx(A):
    n = np.shape(A)[0]
    G = nx.from_numpy_matrix(A)

    return G, n


if __name__ == '__main__':
    from plot_cluster import plot_G
    A = np.array([[0,0,1], [0,0,0], [1,0,0]])
    G, n = np_nx(A)
    print(n)
    plot_G(G)


    G = nx.Graph()
    G.add_node(0)
    G.add_node(1)
    G.add_node(2)
    G.add_edge(0,2,weight=1)
    A, n = nx_np(G)
    print(n)
    print(A)
